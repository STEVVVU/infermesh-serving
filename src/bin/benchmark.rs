use std::collections::BTreeSet;
use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use infermesh::engine::{DeterministicCpuEngine, Dispatcher, EngineDirectory};
use infermesh::metrics::Metrics;
use infermesh::registry::{WorkerRegistration, WorkerRegistry};
use infermesh::repository::ModelRepository;
use infermesh::scheduler::Scheduler;
use infermesh::server::{serve_listener, App};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_lines)]
fn run() -> infermesh::Result<()> {
    let mut arguments = env::args().skip(1);
    let operations = parse_or(arguments.next(), 2_000, "operations")?;
    let concurrency = parse_or(arguments.next(), 16, "concurrency")?;
    if operations == 0 || concurrency == 0 {
        return Err(infermesh::Error::new(
            infermesh::ErrorCode::InvalidArgument,
            "operations and concurrency must be positive",
        ));
    }
    let operations_u32 = u32::try_from(operations).map_err(|_| {
        infermesh::Error::new(
            infermesh::ErrorCode::InvalidArgument,
            "operation count exceeds u32 benchmark reporting limit",
        )
    })?;
    let repository = Arc::new(ModelRepository::open("model_repository")?);
    let registry = Arc::new(WorkerRegistry::new(60_000));
    let engines = Arc::new(EngineDirectory::default());
    let metrics = Arc::new(Metrics::default());
    let models = repository
        .list()
        .into_iter()
        .map(|model| model.key)
        .collect::<BTreeSet<_>>();
    for number in 0..4 {
        let worker_id = format!("benchmark-{number}");
        registry.register(WorkerRegistration {
            id: worker_id.clone(),
            capacity: 4,
            models: models.clone(),
        })?;
        engines.register(worker_id, Arc::new(DeterministicCpuEngine::default()));
    }
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
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    let server = thread::spawn(move || serve_listener(listener, app));

    let next = Arc::new(AtomicUsize::new(0));
    let latencies = Arc::new(Mutex::new(Vec::with_capacity(operations)));
    let errors = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(concurrency + 1));
    let started = Instant::now();
    let mut clients = Vec::new();
    for _ in 0..concurrency {
        let next = Arc::clone(&next);
        let latencies = Arc::clone(&latencies);
        let errors = Arc::clone(&errors);
        let barrier = Arc::clone(&barrier);
        clients.push(thread::spawn(move || {
            barrier.wait();
            loop {
                let request_number = next.fetch_add(1, Ordering::Relaxed);
                if request_number >= operations {
                    return;
                }
                let request_started = Instant::now();
                if infer(address, request_number).is_err() {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
                let latency_us = request_started
                    .elapsed()
                    .as_micros()
                    .try_into()
                    .unwrap_or(u64::MAX);
                latencies
                    .lock()
                    .expect("latency lock poisoned")
                    .push(latency_us);
            }
        }));
    }
    barrier.wait();
    for client in clients {
        client.join().expect("benchmark client did not panic");
    }
    let elapsed = started.elapsed();
    shutdown.store(true, Ordering::SeqCst);
    server
        .join()
        .expect("server did not panic")
        .expect("server stopped cleanly");

    let mut latencies = Arc::try_unwrap(latencies)
        .expect("benchmark owns latency samples")
        .into_inner()
        .expect("latency lock not poisoned");
    latencies.sort_unstable();
    let errors = errors.load(Ordering::Relaxed);
    let successful = operations - errors;
    let successful_f64 = f64::from(
        operations_u32 - u32::try_from(errors).expect("errors are bounded by operations"),
    );
    println!(
        "{}",
        serde_json::json!({
            "benchmark": "triton_v2_http_dynamic_batching",
            "operations": operations,
            "concurrency": concurrency,
            "successful": successful,
            "errors": errors,
            "elapsed_seconds": elapsed.as_secs_f64(),
            "throughput_requests_per_second": successful_f64 / elapsed.as_secs_f64(),
            "latency_us": {
                "p50": percentile(&latencies, 50),
                "p95": percentile(&latencies, 95),
                "p99": percentile(&latencies, 99)
            },
            "model": "adder:1",
            "max_batch_size": 16,
            "max_queue_delay_us": 2000,
            "backend": "deterministic_cpu",
            "gpu_execution": false
        })
    );
    Ok(())
}

fn infer(address: std::net::SocketAddr, request_number: usize) -> std::io::Result<()> {
    let body = serde_json::json!({
        "id": format!("bench-{request_number}"),
        "inputs": [
            {"name": "INPUT0", "shape": [1, 1], "datatype": "FP64", "data": [1.0]},
            {"name": "INPUT1", "shape": [1, 1], "datatype": "FP64", "data": [2.0]}
        ]
    })
    .to_string();
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(5))?;
    write!(
        stream,
        "POST /v2/models/adder/infer HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )?;
    stream.flush()?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    if !response.starts_with("HTTP/1.1 200") {
        return Err(std::io::Error::other("non-200 inference response"));
    }
    Ok(())
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    let index = (sorted.len() * percentile).div_ceil(100).saturating_sub(1);
    sorted[index.min(sorted.len().saturating_sub(1))]
}

fn parse_or(value: Option<String>, default: usize, name: &str) -> infermesh::Result<usize> {
    value.map_or(Ok(default), |value| {
        value.parse().map_err(|error| {
            infermesh::Error::new(
                infermesh::ErrorCode::InvalidArgument,
                format!("invalid {name}: {error}"),
            )
        })
    })
}
