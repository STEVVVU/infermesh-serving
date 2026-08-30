use std::sync::Arc;
use std::time::Duration;

use axum::error_handling::HandleErrorLayer;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{BoxError, Json, Router};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tower::ServiceBuilder;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use crate::metrics::Metrics;
use crate::protocol::{InferRequest, ServerMetadata, MAX_HTTP_BODY_BYTES};
use crate::registry::WorkerRegistry;
use crate::repository::ModelRepository;
use crate::scheduler::Scheduler;
use crate::{Error, ErrorCode, Result};

/// Concurrent in-flight HTTP requests admitted by the service stack.
///
/// This replaces the previous fixed pool of 16 connection threads plus a
/// 256-slot connection queue: with tokio, connections are cheap, so the limit
/// that matters is concurrent *request* work. Exceeding it sheds load with 503
/// rather than queueing without bound.
const MAX_CONCURRENT_REQUESTS: usize = 256;

/// Hard ceiling on how long a single HTTP request may occupy the server.
///
/// Per-request deadlines still come from `timeout_us` or the model config; this
/// only bounds pathological cases such as a stalled request body.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

pub struct App {
    pub repository: Arc<ModelRepository>,
    pub registry: Arc<WorkerRegistry>,
    pub scheduler: Arc<Scheduler>,
    pub metrics: Arc<Metrics>,
    /// Signalled to begin a graceful drain, by Ctrl-C or the control endpoint.
    pub shutdown: Arc<Notify>,
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

/// Control-plane errors rendered as the Triton-style `{"error": ...}` body.
///
/// The status mapping lives on [`ErrorCode`] so every transport agrees.
impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.code.http_status())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        warn!(code = ?self.code, message = %self.message, "request_error");
        (status, Json(json!({"error": self.message}))).into_response()
    }
}

/// Builds the Triton V2 router with the full middleware stack applied.
pub fn router(app: Arc<App>) -> Router {
    // Order matters: tracing wraps everything so shed and timed-out requests
    // are still recorded; the error handler must sit above the fallible
    // middleware it translates.
    let middleware = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(HandleErrorLayer::new(handle_middleware_error))
        .load_shed()
        .concurrency_limit(MAX_CONCURRENT_REQUESTS)
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            REQUEST_TIMEOUT,
        ));

    Router::new()
        .route("/v2", get(server_metadata))
        .route("/v2/health/live", get(health_live))
        .route("/v2/health/ready", get(health_ready))
        .route("/v2/models/{model}", get(model_metadata))
        .route("/v2/models/{model}/ready", get(model_ready))
        .route("/v2/models/{model}/infer", post(infer_latest))
        .route(
            "/v2/models/{model}/versions/{version}/infer",
            post(infer_version),
        )
        .route("/v2/repository/reload", post(repository_reload))
        .route("/v2/control/shutdown", post(control_shutdown))
        .route("/metrics", get(metrics))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(MAX_HTTP_BODY_BYTES))
        .layer(middleware)
        .with_state(app)
}

/// Binds `bind` and serves until shutdown is signalled.
pub async fn serve(bind: &str, app: Arc<App>) -> Result<()> {
    let listener = TcpListener::bind(bind).await?;
    serve_listener(listener, app).await
}

/// Serves on an already-bound listener until shutdown is signalled.
///
/// Tests and the benchmark bind an ephemeral port themselves, so they need this
/// entry point rather than [`serve`].
pub async fn serve_listener(listener: TcpListener, app: Arc<App>) -> Result<()> {
    let bind = listener.local_addr()?;
    info!(
        %bind,
        max_concurrent_requests = MAX_CONCURRENT_REQUESTS,
        "server_started"
    );
    let shutdown = Arc::clone(&app.shutdown);
    let service = router(app);
    axum::serve(listener, service)
        // Graceful shutdown: stop accepting, then let admitted requests finish.
        .with_graceful_shutdown(async move { shutdown.notified().await })
        .await?;
    info!("server_stopped");
    Ok(())
}

/// Translates middleware failures into the same JSON error shape as handlers.
async fn handle_middleware_error(error: BoxError) -> Response {
    if error.is::<tower::load_shed::error::Overloaded>() {
        return Error::new(
            ErrorCode::Unavailable,
            "server is at its concurrent request limit",
        )
        .into_response();
    }
    if error.is::<tower::timeout::error::Elapsed>() {
        return Error::new(ErrorCode::DeadlineExceeded, "request timed out").into_response();
    }
    Error::new(ErrorCode::Internal, error.to_string()).into_response()
}

async fn server_metadata() -> Json<ServerMetadata> {
    Json(ServerMetadata {
        name: "infermesh".to_owned(),
        version: env!("CARGO_PKG_VERSION").to_owned(),
        extensions: vec![
            "model_repository".to_owned(),
            "dynamic_batching".to_owned(),
            "worker_leases".to_owned(),
        ],
    })
}

async fn health_live() -> Json<Value> {
    Json(json!({}))
}

async fn health_ready(State(app): State<Arc<App>>) -> (StatusCode, Json<Value>) {
    let status = if app.ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(json!({})))
}

async fn model_metadata(
    State(app): State<Arc<App>>,
    Path(model): Path<String>,
) -> Result<Json<crate::protocol::ModelMetadata>> {
    Ok(Json(app.repository.metadata(&model)?))
}

async fn model_ready(
    State(app): State<Arc<App>>,
    Path(model): Path<String>,
) -> Result<(StatusCode, Json<Value>)> {
    let loaded = app.repository.resolve(&model, None)?;
    let status = if app.registry.supports_model(&loaded.key) {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    Ok((status, Json(json!({}))))
}

async fn infer_latest(
    State(app): State<Arc<App>>,
    Path(model): Path<String>,
    request: JsonRequest<InferRequest>,
) -> Result<Json<crate::protocol::InferResponse>> {
    Ok(Json(app.scheduler.infer(&model, None, request.0).await?))
}

async fn infer_version(
    State(app): State<Arc<App>>,
    Path((model, version)): Path<(String, String)>,
    request: JsonRequest<InferRequest>,
) -> Result<Json<crate::protocol::InferResponse>> {
    Ok(Json(
        app.scheduler
            .infer(&model, Some(&version), request.0)
            .await?,
    ))
}

async fn repository_reload(State(app): State<Arc<App>>) -> Result<Json<Value>> {
    let generation = app.scheduler.reload()?;
    info!(generation, "repository_reloaded");
    Ok(Json(json!({"generation": generation})))
}

async fn control_shutdown(State(app): State<Arc<App>>) -> (StatusCode, Json<Value>) {
    // `notify_waiters` would race a caller that has not yet awaited, so use the
    // permit-storing variant: shutdown is observed even if requested before the
    // server task reaches its await point.
    app.shutdown.notify_one();
    (StatusCode::ACCEPTED, Json(json!({"shutdown": "draining"})))
}

async fn metrics(State(app): State<Arc<App>>) -> Response {
    let body = app.metrics.render_prometheus(app.registry.live_workers());
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        )],
        body,
    )
        .into_response()
}

async fn not_found() -> Error {
    Error::new(ErrorCode::NotFound, "route not found")
}

/// JSON body extractor that reports malformed input as `InvalidArgument`.
///
/// axum's own `Json` rejection would emit a plain-text 422, which is neither the
/// Triton `{"error": ...}` shape nor the 400 the previous hand-rolled server
/// returned for unparseable bodies.
struct JsonRequest<T>(T);

impl<S, T> axum::extract::FromRequest<S> for JsonRequest<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Error;

    async fn from_request(
        request: axum::extract::Request,
        state: &S,
    ) -> std::result::Result<Self, Self::Rejection> {
        match Json::<T>::from_request(request, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) => Err(Error::new(
                match rejection {
                    // A body over `DefaultBodyLimit` is a resource limit, not a
                    // malformed payload.
                    axum::extract::rejection::JsonRejection::BytesRejection(_) => {
                        ErrorCode::ResourceExhausted
                    }
                    _ => ErrorCode::InvalidArgument,
                },
                rejection.body_text(),
            )),
        }
    }
}
