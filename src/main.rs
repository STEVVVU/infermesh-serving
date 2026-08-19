use std::collections::BTreeSet;
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use infermesh::engine::{
    DeterministicCpuEngine, Dispatcher, Engine, EngineDirectory, ProcessEngine,
};
use infermesh::metrics::Metrics;
use infermesh::registry::{WorkerRegistration, WorkerRegistry};
use infermesh::repository::ModelRepository;
use infermesh::scheduler::Scheduler;
use infermesh::server::{serve, structured_log, App};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> infermesh::Result<()> {
    let mut arguments = env::args().skip(1);
    let command = arguments.next().unwrap_or_else(|| "serve".to_owned());
    if command != "serve" {
        return Err(infermesh::Error::new(
            infermesh::ErrorCode::InvalidArgument,
            "usage: infermesh-server serve [repository] [bind]",
        ));
    }
    let repository_path = arguments
        .next()
        .unwrap_or_else(|| "model_repository".to_owned());
    let bind = arguments
        .next()
        .unwrap_or_else(|| "127.0.0.1:8000".to_owned());
    let repository = Arc::new(ModelRepository::open(repository_path)?);
    let registry = Arc::new(WorkerRegistry::new(15_000));
    let engines = Arc::new(EngineDirectory::default());
    let metrics = Arc::new(Metrics::default());

    let models = repository
        .list()
        .into_iter()
        .map(|model| model.key)
        .collect::<BTreeSet<_>>();
    let engine_mode = env::var("INFERMESH_ENGINE").unwrap_or_else(|_| "rust-cpu".to_owned());
    for worker_id in ["local-cpu-0", "local-cpu-1"] {
        registry.register(WorkerRegistration {
            id: worker_id.to_owned(),
            capacity: 4,
            models: models.clone(),
        })?;
        let engine: Arc<dyn Engine> = match engine_mode.as_str() {
            "python-cpu" => Arc::new(ProcessEngine::spawn(
                "python",
                "cpu",
                Duration::from_secs(30),
            )?),
            "python-onnx" => Arc::new(ProcessEngine::spawn(
                "python",
                "onnxruntime",
                Duration::from_secs(30),
            )?),
            "python-torch" => Arc::new(ProcessEngine::spawn(
                "python",
                "torch",
                Duration::from_secs(30),
            )?),
            _ => Arc::new(DeterministicCpuEngine::default()),
        };
        engines.register(worker_id, engine);
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
    let signal = Arc::clone(&shutdown);
    ctrlc::set_handler(move || {
        signal.store(true, Ordering::SeqCst);
    })
    .map_err(|error| {
        infermesh::Error::new(
            infermesh::ErrorCode::Internal,
            format!("failed to install signal handler: {error}"),
        )
    })?;

    let heartbeat_shutdown = Arc::clone(&shutdown);
    let heartbeat_registry = Arc::clone(&registry);
    let heartbeat = thread::spawn(move || {
        while !heartbeat_shutdown.load(Ordering::SeqCst) {
            for worker_id in ["local-cpu-0", "local-cpu-1"] {
                let _ = heartbeat_registry.heartbeat(worker_id, Vec::new());
            }
            thread::sleep(Duration::from_secs(2));
        }
    });
    let app = Arc::new(App {
        repository,
        registry,
        scheduler,
        metrics,
        shutdown: Arc::clone(&shutdown),
    });
    let result = serve(&bind, app);
    shutdown.store(true, Ordering::SeqCst);
    let _ = heartbeat.join();
    structured_log("shutdown_complete", serde_json::json!({}));
    result
}
