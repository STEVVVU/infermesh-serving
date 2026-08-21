use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use infermesh::engine::{DeterministicCpuEngine, Dispatcher, EngineDirectory};
use infermesh::metrics::Metrics;
use infermesh::registry::{WorkerRegistration, WorkerRegistry};
use infermesh::repository::ModelRepository;
use infermesh::scheduler::Scheduler;
use infermesh::server::{serve_listener, App};
use serde_json::{json, Value};

#[test]
fn triton_v2_endpoints_infer_observe_reload_and_shutdown() {
    let repository = Arc::new(ModelRepository::open("model_repository").unwrap());
    let registry = Arc::new(WorkerRegistry::new(60_000));
    let engines = Arc::new(EngineDirectory::default());
    let metrics = Arc::new(Metrics::default());
    let models = repository
        .list()
        .into_iter()
        .map(|model| model.key)
        .collect::<BTreeSet<_>>();
    registry
        .register(WorkerRegistration {
            id: "integration-worker".to_owned(),
            capacity: 4,
            models,
        })
        .unwrap();
    engines.register(
        "integration-worker",
        Arc::new(DeterministicCpuEngine::default()),
    );
    let dispatcher = Arc::new(Dispatcher::new(
        Arc::clone(&registry),
        engines,
        Arc::clone(&metrics),
    ));
    let scheduler = Arc::new(Scheduler::new(
        Arc::clone(&repository),
        dispatcher,
        Arc::clone(&metrics),
    ));
    let shutdown = Arc::new(AtomicBool::new(false));
    let app = Arc::new(App {
        repository,
        registry,
        scheduler,
        metrics,
        shutdown: Arc::clone(&shutdown),
    });
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || serve_listener(listener, app).unwrap());

    assert_eq!(request(address, "GET", "/v2/health/live", None).0, 200);
    assert_eq!(request(address, "GET", "/v2/health/ready", None).0, 200);
    let metadata = request(address, "GET", "/v2/models/adder", None);
    assert_eq!(metadata.0, 200);
    assert_eq!(metadata.1["name"], json!("adder"));
    assert_eq!(
        request(address, "GET", "/v2/models/adder/ready", None).0,
        200
    );

    let infer = request(
        address,
        "POST",
        "/v2/models/adder/versions/1/infer",
        Some(json!({
            "id": "integration",
            "inputs": [
                {"name": "INPUT0", "shape": [1, 1], "datatype": "FP64", "data": [2.0]},
                {"name": "INPUT1", "shape": [1, 1], "datatype": "FP64", "data": [4.0]}
            ]
        })),
    );
    assert_eq!(infer.0, 200);
    assert_eq!(infer.1["outputs"][0]["data"], json!([6.0]));

    let metrics_response = raw_request(address, "GET", "/metrics", None);
    assert!(metrics_response.starts_with("HTTP/1.1 200"));
    assert!(metrics_response.contains("infermesh_requests_total 1"));
    assert_eq!(
        request(address, "POST", "/v2/repository/reload", Some(json!({}))).0,
        200
    );
    assert_eq!(
        request(address, "POST", "/v2/control/shutdown", Some(json!({}))).0,
        202
    );
    server.join().unwrap();
    assert!(shutdown.load(Ordering::SeqCst));
}

fn request(address: SocketAddr, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
    let response = raw_request(address, method, path, body);
    let (header, body) = response.split_once("\r\n\r\n").unwrap();
    let status = header.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(body).unwrap())
}

fn raw_request(address: SocketAddr, method: &str, path: &str, body: Option<Value>) -> String {
    let body = body.map_or_else(String::new, |value| value.to_string());
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(5)).unwrap();
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    stream.flush().unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response
}
