use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use infermesh::engine::{
    DeterministicCpuEngine, Dispatcher, Engine, EngineDirectory, ProcessEngine,
};
use infermesh::metrics::Metrics;
use infermesh::registry::{WorkerRegistration, WorkerRegistry};
use infermesh::repository::ModelRepository;
use infermesh::scheduler::Scheduler;
use infermesh::server::{serve, App};
use tokio::sync::Notify;
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Worker execution transport.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum EngineMode {
    /// In-process deterministic Rust CPU engine.
    #[default]
    RustCpu,
    /// Python worker subprocess using the dependency-free CPU adapter.
    PythonCpu,
    /// Python worker subprocess using the ONNX Runtime CPU adapter.
    PythonOnnx,
    /// Python worker subprocess using the optional `TorchScript` CPU adapter.
    PythonTorch,
}

/// Log output format.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum LogFormat {
    /// One JSON object per event, for log shipping.
    #[default]
    Json,
    /// Human-readable console output.
    Text,
}

#[derive(Debug, Parser)]
#[command(
    name = "infermesh-server",
    version,
    about = "InferMesh serving control plane"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the Triton V2 HTTP API.
    Serve(ServeArgs),
}

#[derive(Debug, Parser)]
struct ServeArgs {
    /// Filesystem model repository root.
    #[arg(default_value = "model_repository")]
    repository: PathBuf,

    /// Address to bind the HTTP server to.
    #[arg(default_value = "127.0.0.1:8000")]
    bind: String,

    /// Worker execution transport.
    ///
    /// `INFERMESH_ENGINE` is honoured for compatibility with earlier revisions.
    #[arg(long, value_enum, default_value_t, env = "INFERMESH_ENGINE")]
    engine: EngineMode,

    /// Log output format.
    #[arg(long, value_enum, default_value_t)]
    log_format: LogFormat,

    /// Number of local workers to register.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..))]
    workers: u8,

    /// Concurrent execution slots per worker.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u16).range(1..))]
    worker_capacity: u16,

    /// Heartbeat lease duration in milliseconds.
    #[arg(long, default_value_t = 15_000)]
    lease_ms: u64,

    /// Python interpreter used for subprocess engines.
    #[arg(long, default_value = "python")]
    python: String,

    /// Per-call timeout for Python subprocess engines, in seconds.
    #[arg(long, default_value_t = 30)]
    engine_timeout_seconds: u64,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let Command::Serve(arguments) = cli.command;
    init_tracing(arguments.log_format);

    // The runtime is built explicitly rather than via `#[tokio::main]` so
    // tracing is initialised before any runtime thread starts and so startup
    // failures are reported through `anyhow` at the binary boundary.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build the Tokio runtime")?
        .block_on(run(arguments))
}

fn init_tracing(format: LogFormat) {
    // Default to INFO; `RUST_LOG` overrides per-target as usual.
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("infermesh=info,info"));
    let registry = tracing_subscriber::registry().with(filter);
    match format {
        LogFormat::Json => registry
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_current_span(true),
            )
            .init(),
        LogFormat::Text => registry.with(tracing_subscriber::fmt::layer()).init(),
    }
}

async fn run(arguments: ServeArgs) -> anyhow::Result<()> {
    let repository = Arc::new(
        ModelRepository::open(&arguments.repository).with_context(|| {
            format!(
                "failed to open model repository {}",
                arguments.repository.display()
            )
        })?,
    );
    let registry = Arc::new(WorkerRegistry::new(arguments.lease_ms));
    let engines = Arc::new(EngineDirectory::default());
    let metrics = Arc::new(Metrics::default());

    let models = repository
        .list()
        .into_iter()
        .map(|model| model.key)
        .collect::<BTreeSet<_>>();
    let worker_ids = (0..arguments.workers)
        .map(|number| format!("local-cpu-{number}"))
        .collect::<Vec<_>>();
    let engine_timeout = Duration::from_secs(arguments.engine_timeout_seconds);
    for worker_id in &worker_ids {
        registry
            .register(WorkerRegistration {
                id: worker_id.clone(),
                capacity: usize::from(arguments.worker_capacity),
                models: models.clone(),
            })
            .context("failed to register a local worker")?;
        let engine: Arc<dyn Engine> = match arguments.engine {
            EngineMode::RustCpu => Arc::new(DeterministicCpuEngine::default()),
            EngineMode::PythonCpu => Arc::new(
                ProcessEngine::spawn(&arguments.python, "cpu", engine_timeout)
                    .context("failed to start the Python CPU worker")?,
            ),
            EngineMode::PythonOnnx => Arc::new(
                ProcessEngine::spawn(&arguments.python, "onnxruntime", engine_timeout)
                    .context("failed to start the Python ONNX Runtime worker")?,
            ),
            EngineMode::PythonTorch => Arc::new(
                ProcessEngine::spawn(&arguments.python, "torch", engine_timeout)
                    .context("failed to start the Python Torch worker")?,
            ),
        };
        engines.register(worker_id.clone(), engine);
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

    // Ctrl-C triggers the same graceful drain as the control endpoint.
    let signal_shutdown = Arc::clone(&shutdown);
    tokio::spawn(async move {
        match tokio::signal::ctrl_c().await {
            Ok(()) => {
                info!("shutdown_requested");
                signal_shutdown.notify_one();
            }
            Err(error) => {
                tracing::error!(%error, "failed to install the Ctrl-C handler");
            }
        }
    });

    // Local workers have no out-of-process heartbeat, so the control plane
    // renews their leases on their behalf.
    let heartbeat = tokio::spawn(heartbeat_loop(
        Arc::clone(&registry),
        worker_ids,
        arguments.lease_ms,
    ));

    let app = Arc::new(App {
        repository,
        registry,
        scheduler,
        metrics,
        shutdown: Arc::clone(&shutdown),
    });
    let result = serve(&arguments.bind, app).await;
    heartbeat.abort();
    info!("shutdown_complete");
    result.map_err(anyhow::Error::new)
}

async fn heartbeat_loop(registry: Arc<WorkerRegistry>, worker_ids: Vec<String>, lease_ms: u64) {
    // Renew well inside the lease so a scheduling hiccup cannot expire a
    // healthy local worker.
    let period = Duration::from_millis((lease_ms / 4).max(1));
    let mut ticker = tokio::time::interval(period);
    loop {
        ticker.tick().await;
        for worker_id in &worker_ids {
            let _ = registry.heartbeat(worker_id, Vec::new());
        }
    }
}
