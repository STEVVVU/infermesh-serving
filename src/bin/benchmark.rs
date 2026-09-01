use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Parser;
use infermesh::engine::{DeterministicCpuEngine, Dispatcher, EngineDirectory};
use infermesh::metrics::Metrics;
use infermesh::registry::{WorkerRegistration, WorkerRegistry};
use infermesh::repository::ModelRepository;
use infermesh::scheduler::Scheduler;
use infermesh::server::{serve_listener, App};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Barrier, Mutex, Notify};

#[derive(Debug, Parser)]
#[command(
    name = "infermesh-bench",
    version,
    about = "End-to-end Triton V2 HTTP load benchmark"
)]
struct Cli {
    /// Total inference requests to send.
    #[arg(default_value_t = 2_000)]
    operations: usize,

    /// Concurrent client tasks.
    #[arg(default_value_t = 16)]
    concurrency: usize,

    /// Deterministic CPU workers to register.
    #[arg(long, default_value_t = 4)]
    workers: usize,

    /// Concurrent execution slots per worker.
    #[arg(long, default_value_t = 4)]
    worker_capacity: usize,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    // The server and the load generator get separate runtimes on purpose.
    //
    // Sharing one runtime makes the client tasks compete with the server's own
    // worker threads for the same cores, which shows up as server tail latency
    // even though it is really client scheduling delay. The previous
    // thread-based harness gave clients their own OS threads and so did not
    // have this problem; splitting the runtimes preserves that property and
    // keeps the measurement about the server.
    let server_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build the server runtime")?;
    let server = server_runtime.block_on(start_server(&cli))?;
    let client_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build the client runtime")?;
    client_runtime.block_on(run(cli, server))
}

/// The ephemeral server under test, plus the handles needed to stop it.
struct Server {
    address: SocketAddr,
    shutdown: Arc<Notify>,
    task: tokio::task::JoinHandle<infermesh::Result<()>>,
}

/// Builds the repository, workers, scheduler, and server the benchmark drives.
async fn start_server(cli: &Cli) -> anyhow::Result<Server> {
    let repository = Arc::new(ModelRepository::open("model_repository")?);
    let registry = Arc::new(WorkerRegistry::new(60_000));
    let engines = Arc::new(EngineDirectory::default());
    let metrics = Arc::new(Metrics::default());
    let models = repository
        .list()
        .into_iter()
        .map(|model| model.key)
        .collect::<BTreeSet<_>>();
    for number in 0..cli.workers {
        let worker_id = format!("benchmark-{number}");
        registry.register(WorkerRegistration {
            id: worker_id.clone(),
            capacity: cli.worker_capacity,
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
    let shutdown = Arc::new(Notify::new());
    let app = Arc::new(App {
        repository,
        registry,
        scheduler,
        metrics,
        shutdown: Arc::clone(&shutdown),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let task = tokio::spawn(serve_listener(listener, app));
    Ok(Server {
        address,
        shutdown,
        task,
    })
}

async fn run(cli: Cli, server: Server) -> anyhow::Result<()> {
    anyhow::ensure!(
        cli.operations > 0 && cli.concurrency > 0,
        "operations and concurrency must be positive"
    );
    anyhow::ensure!(
        cli.workers > 0 && cli.worker_capacity > 0,
        "workers and worker capacity must be positive"
    );
    let operations_u32 = u32::try_from(cli.operations)
        .context("operation count exceeds u32 benchmark reporting limit")?;

    let address = server.address;

    let next = Arc::new(AtomicUsize::new(0));
    let latencies = Arc::new(Mutex::new(Vec::with_capacity(cli.operations)));
    let errors = Arc::new(AtomicUsize::new(0));
    // Clients start together so the measured window excludes task spawn cost.
    let barrier = Arc::new(Barrier::new(cli.concurrency + 1));
    let mut clients = Vec::with_capacity(cli.concurrency);
    for _ in 0..cli.concurrency {
        let next = Arc::clone(&next);
        let latencies = Arc::clone(&latencies);
        let errors = Arc::clone(&errors);
        let barrier = Arc::clone(&barrier);
        let operations = cli.operations;
        clients.push(tokio::spawn(async move {
            barrier.wait().await;
            let mut samples = Vec::new();
            loop {
                let request_number = next.fetch_add(1, Ordering::Relaxed);
                if request_number >= operations {
                    break;
                }
                let request_started = Instant::now();
                if infer(address, request_number).await.is_err() {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
                samples.push(
                    request_started
                        .elapsed()
                        .as_micros()
                        .try_into()
                        .unwrap_or(u64::MAX),
                );
            }
            // Samples are merged once per client rather than per request, so
            // the shared lock is not part of the measured path.
            latencies.lock().await.extend(samples);
        }));
    }
    barrier.wait().await;
    let started = Instant::now();
    for client in clients {
        client.await.context("benchmark client task panicked")?;
    }
    let elapsed = started.elapsed();
    server.shutdown.notify_one();
    server
        .task
        .await
        .context("server task panicked")?
        .context("server stopped with an error")?;

    let mut latencies = latencies.lock().await;
    latencies.sort_unstable();
    let errors = errors.load(Ordering::Relaxed);
    let successful = cli.operations - errors;
    let successful_f64 = f64::from(
        operations_u32 - u32::try_from(errors).expect("errors are bounded by operations"),
    );
    println!(
        "{}",
        serde_json::json!({
            "benchmark": "triton_v2_http_dynamic_batching",
            "operations": cli.operations,
            "concurrency": cli.concurrency,
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
            "runtime": "tokio",
            "gpu_execution": false
        })
    );
    Ok(())
}

async fn infer(address: SocketAddr, request_number: usize) -> std::io::Result<()> {
    let body = serde_json::json!({
        "id": format!("bench-{request_number}"),
        "inputs": [
            {"name": "INPUT0", "shape": [1, 1], "datatype": "FP64", "data": [1.0]},
            {"name": "INPUT1", "shape": [1, 1], "datatype": "FP64", "data": [2.0]}
        ]
    })
    .to_string();
    let mut stream = tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(address))
        .await
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "connect timed out"))??;
    let request = format!(
        "POST /v2/models/adder/infer HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await?;
    stream.flush().await?;
    let mut response = String::new();
    stream.read_to_string(&mut response).await?;
    if !response.starts_with("HTTP/1.1 200") {
        return Err(std::io::Error::other("non-200 inference response"));
    }
    Ok(())
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let index = (sorted.len() * percentile).div_ceil(100).saturating_sub(1);
    sorted[index.min(sorted.len().saturating_sub(1))]
}
