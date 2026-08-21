use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};

use crate::metrics::Metrics;
use crate::protocol::{InferRequest, ServerMetadata, MAX_HTTP_BODY_BYTES};
use crate::registry::WorkerRegistry;
use crate::repository::ModelRepository;
use crate::scheduler::Scheduler;
use crate::{Error, ErrorCode, Result};

const MAX_HEADER_BYTES: usize = 64 * 1024;
const CONNECTION_QUEUE: usize = 256;
const HTTP_WORKERS: usize = 16;

pub struct App {
    pub repository: Arc<ModelRepository>,
    pub registry: Arc<WorkerRegistry>,
    pub scheduler: Arc<Scheduler>,
    pub metrics: Arc<Metrics>,
    pub shutdown: Arc<AtomicBool>,
}

impl App {
    fn ready(&self) -> bool {
        let models = self.repository.list();
        !models.is_empty()
            && models
                .iter()
                .all(|model| self.registry.supports_model(&model.key))
    }
}

pub fn serve(bind: &str, app: Arc<App>) -> Result<()> {
    let listener = TcpListener::bind(bind)?;
    serve_listener(listener, app)
}

pub fn serve_listener(listener: TcpListener, app: Arc<App>) -> Result<()> {
    let bind = listener.local_addr()?.to_string();
    listener.set_nonblocking(true)?;
    structured_log(
        "server_started",
        json!({"bind": bind, "http_workers": HTTP_WORKERS}),
    );
    let (sender, receiver) = mpsc::sync_channel(CONNECTION_QUEUE);
    let receiver = Arc::new(Mutex::new(receiver));
    let mut workers = Vec::new();
    for number in 0..HTTP_WORKERS {
        let receiver = Arc::clone(&receiver);
        let app = Arc::clone(&app);
        workers.push(
            thread::Builder::new()
                .name(format!("http-{number}"))
                .spawn(move || connection_worker(receiver, app))
                .expect("HTTP worker creation succeeds"),
        );
    }

    while !app.shutdown.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false)?;
                admit_connection(&sender, stream);
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(error) => return Err(error.into()),
        }
    }
    drop(sender);
    for worker in workers {
        let _ = worker.join();
    }
    structured_log("server_stopped", json!({}));
    Ok(())
}

fn admit_connection(sender: &SyncSender<TcpStream>, stream: TcpStream) {
    match sender.try_send(stream) {
        Ok(()) | Err(TrySendError::Disconnected(_)) => {}
        Err(TrySendError::Full(mut stream)) => {
            let _ = write_response(
                &mut stream,
                HttpResponse::json(503, json!({"error": "HTTP connection queue is full"})),
            );
        }
    }
}

fn connection_worker(receiver: Arc<Mutex<mpsc::Receiver<TcpStream>>>, app: Arc<App>) {
    loop {
        let stream = receiver
            .lock()
            .expect("connection receiver lock poisoned")
            .recv();
        let Ok(mut stream) = stream else {
            return;
        };
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let response = read_request(&mut stream)
            .and_then(|request| route(&app, request))
            .unwrap_or_else(error_response);
        let _ = write_response(&mut stream, response);
    }
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

struct HttpResponse {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
}

impl HttpResponse {
    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: serde_json::to_vec(&value).expect("JSON value serializes"),
        }
    }

    fn serializable(status: u16, value: &impl Serialize) -> Result<Self> {
        Ok(Self {
            status,
            content_type: "application/json",
            body: serde_json::to_vec(value)?,
        })
    }

    fn text(status: u16, content_type: &'static str, body: String) -> Self {
        Self {
            status,
            content_type,
            body: body.into_bytes(),
        }
    }
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let mut received = Vec::new();
    let mut buffer = [0_u8; 4096];
    let header_end = loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "connection closed before HTTP headers",
            ));
        }
        received.extend_from_slice(&buffer[..count]);
        if received.len() > MAX_HEADER_BYTES + MAX_HTTP_BODY_BYTES {
            return Err(Error::new(
                ErrorCode::ResourceExhausted,
                "HTTP request exceeds configured limits",
            ));
        }
        if let Some(position) = find_bytes(&received, b"\r\n\r\n") {
            break position + 4;
        }
        if received.len() > MAX_HEADER_BYTES {
            return Err(Error::new(
                ErrorCode::ResourceExhausted,
                "HTTP headers exceed configured limit",
            ));
        }
    };
    let header = std::str::from_utf8(&received[..header_end]).map_err(|_| {
        Error::new(
            ErrorCode::InvalidArgument,
            "HTTP headers must be valid UTF-8",
        )
    })?;
    let mut lines = header.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| Error::new(ErrorCode::InvalidArgument, "missing HTTP request line"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or_default().to_owned();
    if method.is_empty() || path.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "malformed HTTP request line",
        ));
    }
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_owned()))
        .collect::<BTreeMap<_, _>>();
    if headers
        .get("transfer-encoding")
        .is_some_and(|value| value != "identity")
    {
        return Err(Error::new(
            ErrorCode::InvalidArgument,
            "chunked transfer encoding is not supported",
        ));
    }
    let content_length = headers.get("content-length").map_or(Ok(0), |value| {
        value.parse::<usize>().map_err(|error| {
            Error::new(
                ErrorCode::InvalidArgument,
                format!("invalid Content-Length: {error}"),
            )
        })
    })?;
    if content_length > MAX_HTTP_BODY_BYTES {
        return Err(Error::new(
            ErrorCode::ResourceExhausted,
            "HTTP body exceeds configured limit",
        ));
    }
    while received.len() - header_end < content_length {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Err(Error::new(
                ErrorCode::InvalidArgument,
                "HTTP body ended before Content-Length",
            ));
        }
        received.extend_from_slice(&buffer[..count]);
    }
    Ok(HttpRequest {
        method,
        path,
        body: received[header_end..header_end + content_length].to_vec(),
    })
}

fn route(app: &App, request: HttpRequest) -> Result<HttpResponse> {
    let path = request.path.split('?').next().unwrap_or(&request.path);
    if request.method == "GET" && path == "/v2/health/live" {
        return Ok(HttpResponse::json(200, json!({})));
    }
    if request.method == "GET" && path == "/v2/health/ready" {
        return Ok(HttpResponse::json(
            if app.ready() { 200 } else { 503 },
            json!({}),
        ));
    }
    if request.method == "GET" && path == "/v2" {
        return HttpResponse::serializable(
            200,
            &ServerMetadata {
                name: "infermesh".to_owned(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
                extensions: vec![
                    "model_repository".to_owned(),
                    "dynamic_batching".to_owned(),
                    "worker_leases".to_owned(),
                ],
            },
        );
    }
    if request.method == "GET" && path == "/metrics" {
        return Ok(HttpResponse::text(
            200,
            "text/plain; version=0.0.4",
            app.metrics.render_prometheus(app.registry.live_workers()),
        ));
    }
    if request.method == "POST" && path == "/v2/repository/reload" {
        let generation = app.scheduler.reload()?;
        structured_log("repository_reloaded", json!({"generation": generation}));
        return Ok(HttpResponse::json(200, json!({"generation": generation})));
    }
    if request.method == "POST" && path == "/v2/control/shutdown" {
        app.shutdown.store(true, Ordering::SeqCst);
        return Ok(HttpResponse::json(202, json!({"shutdown": "draining"})));
    }
    route_model(app, &request.method, path, &request.body)
}

fn route_model(app: &App, method: &str, path: &str, body: &[u8]) -> Result<HttpResponse> {
    let parts = path.trim_matches('/').split('/').collect::<Vec<_>>();
    if parts.len() < 3 || parts[0] != "v2" || parts[1] != "models" {
        return Err(Error::new(ErrorCode::NotFound, "route not found"));
    }
    let model = parts[2];
    if method == "GET" && parts.len() == 4 && parts[3] == "ready" {
        let loaded = app.repository.resolve(model, None)?;
        let status = if app.registry.supports_model(&loaded.key) {
            200
        } else {
            503
        };
        return Ok(HttpResponse::json(status, json!({})));
    }
    if method == "GET" && parts.len() == 3 {
        return HttpResponse::serializable(200, &app.repository.metadata(model)?);
    }

    let (version, infer) = match parts.as_slice() {
        [_, _, _, "infer"] => (None, true),
        [_, _, _, "versions", version, "infer"] => (Some(*version), true),
        _ => (None, false),
    };
    if method == "POST" && infer {
        let infer_request: InferRequest = serde_json::from_slice(body)?;
        let response = app.scheduler.infer(model, version, infer_request)?;
        return HttpResponse::serializable(200, &response);
    }
    Err(Error::new(ErrorCode::NotFound, "route not found"))
}

fn error_response(error: Error) -> HttpResponse {
    let status = match error.code {
        ErrorCode::InvalidArgument => 400,
        ErrorCode::NotFound => 404,
        ErrorCode::ResourceExhausted => 429,
        ErrorCode::DeadlineExceeded => 408,
        ErrorCode::Unavailable => 503,
        ErrorCode::Internal => 500,
    };
    structured_log(
        "request_error",
        json!({"code": format!("{:?}", error.code), "message": error.message}),
    );
    HttpResponse::json(status, json!({"error": error.message}))
}

fn write_response(stream: &mut TcpStream, response: HttpResponse) -> std::io::Result<()> {
    let reason = match response.status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        404 => "Not Found",
        408 => "Request Timeout",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Response",
    };
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        reason,
        response.content_type,
        response.body.len()
    )?;
    stream.write_all(&response.body)?;
    stream.flush()
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub fn structured_log(event: &str, fields: Value) {
    eprintln!(
        "{}",
        json!({
            "ts_unix_ms": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            "level": "INFO",
            "event": event,
            "fields": fields
        })
    );
}
