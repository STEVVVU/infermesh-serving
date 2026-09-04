//! End-to-end Triton V2 HTTP tests.
//!
//! These deliberately speak raw HTTP/1.1 over a real TCP socket rather than
//! calling the router in-process, so the wire contract — status lines, headers,
//! and JSON bodies — is what gets asserted.

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use infermesh::engine::{DeterministicCpuEngine, Dispatcher, EngineDirectory};
use infermesh::metrics::Metrics;
use infermesh::registry::{WorkerRegistration, WorkerRegistry};
use infermesh::repository::ModelRepository;
use infermesh::scheduler::Scheduler;
use infermesh::server::{serve_listener, App};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

struct Harness {
    address: SocketAddr,
    shutdown: Arc<Notify>,
    metrics: Arc<Metrics>,
    server: JoinHandle<infermesh::Result<()>>,
}

/// Per-request `timeout_us` for inferences whose *success* is asserted.
///
/// The checked-in `adder` config carries a 500 ms `default_timeout_us` — a
/// sensible production value, but it means every success assertion below would
/// otherwise ride on the request finishing inside 500 ms. That is real
/// fragility: on a loaded machine (shared CI runner, debug build, tests running
/// in parallel) queueing plus execution overruns it and the server correctly
/// answers 408, failing a test that was never about timeliness. These tests opt
/// out through the wire's own `timeout_us` parameter, which also keeps the
/// override visible in the request rather than hidden in the model config.
///
/// The deadline path itself is asserted deliberately, with a 1 ms deadline
/// against a 50 ms engine, in
/// `request_timeout_parameter_returns_request_timeout_status`.
const UNREACHABLE_TIMEOUT_US: u64 = 60_000_000;

async fn start() -> Harness {
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
    let shutdown = Arc::new(Notify::new());
    let app = Arc::new(App {
        repository,
        registry,
        scheduler,
        metrics: Arc::clone(&metrics),
        shutdown: Arc::clone(&shutdown),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_listener(listener, app));
    Harness {
        address,
        shutdown,
        metrics,
        server,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn triton_v2_endpoints_infer_observe_reload_and_shutdown() {
    let harness = start().await;
    let address = harness.address;

    assert_eq!(
        request(address, "GET", "/v2/health/live", None).await.0,
        200
    );
    assert_eq!(
        request(address, "GET", "/v2/health/ready", None).await.0,
        200
    );
    let metadata = request(address, "GET", "/v2/models/adder", None).await;
    assert_eq!(metadata.0, 200);
    assert_eq!(metadata.1["name"], json!("adder"));
    assert_eq!(
        request(address, "GET", "/v2/models/adder/ready", None)
            .await
            .0,
        200
    );

    let infer = request(
        address,
        "POST",
        "/v2/models/adder/versions/1/infer",
        Some(json!({
            "id": "integration",
            "parameters": {"timeout_us": UNREACHABLE_TIMEOUT_US},
            "inputs": [
                {"name": "INPUT0", "shape": [1, 1], "datatype": "FP64", "data": [2.0]},
                {"name": "INPUT1", "shape": [1, 1], "datatype": "FP64", "data": [4.0]}
            ]
        })),
    )
    .await;
    assert_eq!(infer.0, 200);
    assert_eq!(infer.1["outputs"][0]["data"], json!([6.0]));

    let metrics_response = raw_request(address, "GET", "/metrics", None).await;
    assert!(metrics_response.starts_with("HTTP/1.1 200"));
    assert!(metrics_response.contains("infermesh_requests_total 1"));
    assert_eq!(
        request(address, "POST", "/v2/repository/reload", Some(json!({})))
            .await
            .0,
        200
    );
    assert_eq!(
        request(address, "POST", "/v2/control/shutdown", Some(json!({})))
            .await
            .0,
        202
    );
    harness.server.await.unwrap().unwrap();
}

/// Wire-shape checks for the routes and payloads that must keep their contract.
#[tokio::test(flavor = "multi_thread")]
async fn triton_v2_contract_details_are_preserved() {
    let harness = start().await;
    let address = harness.address;

    // Server metadata advertises the documented extensions.
    let metadata = request(address, "GET", "/v2", None).await;
    assert_eq!(metadata.0, 200);
    assert_eq!(metadata.1["name"], json!("infermesh"));
    assert_eq!(
        metadata.1["extensions"],
        json!(["model_repository", "dynamic_batching", "worker_leases"])
    );

    // Latest-version inference resolves without an explicit version.
    let latest = request(
        address,
        "POST",
        "/v2/models/adder/infer",
        Some(json!({
            "id": "latest",
            "parameters": {"timeout_us": UNREACHABLE_TIMEOUT_US},
            "inputs": [
                {"name": "INPUT0", "shape": [1, 1], "datatype": "FP64", "data": [1.0]},
                {"name": "INPUT1", "shape": [1, 1], "datatype": "FP64", "data": [1.0]}
            ]
        })),
    )
    .await;
    assert_eq!(latest.0, 200);
    assert_eq!(latest.1["model_name"], json!("adder"));
    assert_eq!(latest.1["model_version"], json!("1"));
    assert_eq!(latest.1["id"], json!("latest"));
    assert_eq!(latest.1["outputs"][0]["data"], json!([2.0]));

    // Unknown models are 404 with the `{"error": ...}` body.
    let missing = request(address, "GET", "/v2/models/nonexistent", None).await;
    assert_eq!(missing.0, 404);
    assert!(missing.1["error"].is_string());

    // Unknown routes are 404 rather than a framework-shaped rejection.
    let unrouted = request(address, "GET", "/v2/does-not-exist", None).await;
    assert_eq!(unrouted.0, 404);
    assert_eq!(unrouted.1["error"], json!("route not found"));

    // A schema-invalid request is rejected before scheduling.
    let invalid = request(
        address,
        "POST",
        "/v2/models/adder/infer",
        Some(json!({"id": "bad", "inputs": []})),
    )
    .await;
    assert_eq!(invalid.0, 400);
    assert!(invalid.1["error"].is_string());

    // Unparseable JSON is a 400, not a framework 422.
    let malformed = raw_request_body(
        address,
        "POST",
        "/v2/models/adder/infer",
        "{not json".to_owned(),
    )
    .await;
    assert!(malformed.starts_with("HTTP/1.1 400"));

    // Query strings are ignored on health routes, as before.
    assert_eq!(
        request(address, "GET", "/v2/health/live?verbose=1", None)
            .await
            .0,
        200
    );

    // The metrics endpoint keeps the Prometheus text content type.
    let metrics_response = raw_request(address, "GET", "/metrics", None).await;
    assert!(metrics_response.contains("text/plain; version=0.0.4"));
    assert!(metrics_response.contains("infermesh_live_workers 1"));

    harness.shutdown.notify_one();
    harness.server.await.unwrap().unwrap();
}

/// A per-request `timeout_us` must be honoured end to end over HTTP.
#[tokio::test(flavor = "multi_thread")]
async fn request_timeout_parameter_returns_request_timeout_status() {
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
            id: "slow-worker".to_owned(),
            capacity: 1,
            models,
        })
        .unwrap();
    // 50 ms of engine work against a 1 ms client deadline.
    engines.register(
        "slow-worker",
        Arc::new(DeterministicCpuEngine::with_delay(50_000)),
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
    let shutdown = Arc::new(Notify::new());
    let app = Arc::new(App {
        repository,
        registry,
        scheduler,
        metrics: Arc::clone(&metrics),
        shutdown: Arc::clone(&shutdown),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_listener(listener, app));

    let response = request(
        address,
        "POST",
        "/v2/models/adder/infer",
        Some(json!({
            "id": "too-slow",
            "parameters": {"timeout_us": 1000},
            "inputs": [
                {"name": "INPUT0", "shape": [1, 1], "datatype": "FP64", "data": [1.0]},
                {"name": "INPUT1", "shape": [1, 1], "datatype": "FP64", "data": [2.0]}
            ]
        })),
    )
    .await;
    assert_eq!(response.0, 408);
    assert!(response.1["error"].is_string());
    assert_eq!(metrics.deadline_exceeded_total.get(), 1);

    // The server is still healthy after shedding the timed-out request.
    assert_eq!(
        request(address, "GET", "/v2/health/live", None).await.0,
        200
    );

    shutdown.notify_one();
    server.await.unwrap().unwrap();
}

/// Concurrent load must not break batching accounting or the wire contract.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_requests_are_all_answered_correctly() {
    let harness = start().await;
    let address = harness.address;

    let mut tasks = Vec::new();
    for number in 0..32_u32 {
        tasks.push(tokio::spawn(async move {
            request(
                address,
                "POST",
                "/v2/models/adder/infer",
                Some(json!({
                    "id": format!("concurrent-{number}"),
                    "parameters": {"timeout_us": UNREACHABLE_TIMEOUT_US},
                    "inputs": [
                        {"name": "INPUT0", "shape": [1, 1], "datatype": "FP64", "data": [number]},
                        {"name": "INPUT1", "shape": [1, 1], "datatype": "FP64", "data": [1.0]}
                    ]
                })),
            )
            .await
        }));
    }
    for (number, task) in tasks.into_iter().enumerate() {
        let (status, body) = task.await.unwrap();
        assert_eq!(status, 200);
        // Each response carries its own request's id and result, so responses
        // are never crossed between concurrent tasks.
        assert_eq!(body["id"], json!(format!("concurrent-{number}")));
        assert_eq!(
            body["outputs"][0]["data"],
            json!([f64::from(u32::try_from(number).unwrap()) + 1.0])
        );
    }

    harness.shutdown.notify_one();
    harness.server.await.unwrap().unwrap();
}

/// Graceful shutdown must let an already-accepted request finish.
///
/// The connection is established and the request written *before* shutdown is
/// signalled, because a graceful drain deliberately stops accepting new
/// connections — a connect issued after the signal is expected to be refused.
#[tokio::test(flavor = "multi_thread")]
async fn graceful_shutdown_drains_an_in_flight_request() {
    let harness = start().await;

    let body = json!({
        "id": "draining",
        "parameters": {"timeout_us": UNREACHABLE_TIMEOUT_US},
        "inputs": [
            {"name": "INPUT0", "shape": [1, 1], "datatype": "FP64", "data": [3.0]},
            {"name": "INPUT1", "shape": [1, 1], "datatype": "FP64", "data": [4.0]}
        ]
    })
    .to_string();
    let mut stream = TcpStream::connect(harness.address).await.unwrap();
    let request = format!(
        "POST /v2/models/adder/infer HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();

    // A completed client-side connect only means the kernel finished the
    // handshake into the accept backlog — it does not mean axum has accepted
    // the connection yet. Signalling the drain before that accept happens is a
    // legitimate race that resets the socket, so wait for the server to have
    // actually taken ownership of the request. `requests_total` is incremented
    // once the handler reaches the scheduler, which is strictly after accept.
    await_metric(&harness.metrics, |metrics| {
        metrics.requests_total.get() == 1
    })
    .await;

    // The request is now owned by the server; ask it to drain.
    harness.shutdown.notify_one();

    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    let (header, response_body) = response.split_once("\r\n\r\n").unwrap();
    assert!(header.starts_with("HTTP/1.1 200"));
    let parsed: Value = serde_json::from_str(response_body).unwrap();
    assert_eq!(parsed["outputs"][0]["data"], json!([7.0]));

    harness.server.await.unwrap().unwrap();
}

/// Polls until `predicate` holds, so tests synchronize on observed server state
/// rather than on a sleep long enough to "probably" be sufficient.
async fn await_metric(metrics: &Metrics, predicate: impl Fn(&Metrics) -> bool) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !predicate(metrics) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "server state never reached the expected value within 10s"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

async fn request(
    address: SocketAddr,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> (u16, Value) {
    let response = raw_request(address, method, path, body).await;
    let (header, body) = response.split_once("\r\n\r\n").unwrap();
    let status = header.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, serde_json::from_str(body).unwrap())
}

async fn raw_request(address: SocketAddr, method: &str, path: &str, body: Option<Value>) -> String {
    let body = body.map_or_else(String::new, |value| value.to_string());
    raw_request_body(address, method, path, body).await
}

async fn raw_request_body(address: SocketAddr, method: &str, path: &str, body: String) -> String {
    let mut stream = tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(address))
        .await
        .unwrap()
        .unwrap();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    response
}
