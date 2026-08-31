//! Criterion benchmarks for the scheduling hot paths.
//!
//! These measure the control-plane work that happens per request *inside* the
//! process, with no sockets involved: shape-signature computation and admission
//! validation (per request), worker selection (per batch), and prefix-cache
//! acquisition (per LLM request). The end-to-end HTTP number lives in the
//! `infermesh-bench` binary instead — the two answer different questions and
//! neither substitutes for the other.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use infermesh::engine::{DeterministicCpuEngine, Dispatcher, EngineDirectory};
use infermesh::kv_cache::PagedKvCache;
use infermesh::metrics::Metrics;
use infermesh::protocol::{InferRequest, InferTensor, TensorMetadata};
use infermesh::registry::{WorkerRegistration, WorkerRegistry};
use infermesh::repository::{LoadedModel, ModelConfig, ModelKey};
use infermesh::scheduler::Scheduler;
use serde_json::json;

fn model(name: &str, max_batch_size: usize, max_queue_delay_us: u64) -> LoadedModel {
    LoadedModel {
        key: ModelKey {
            name: name.to_owned(),
            version: "1".to_owned(),
        },
        config: ModelConfig {
            name: name.to_owned(),
            platform: "cpu".to_owned(),
            backend: "deterministic_cpu".to_owned(),
            max_batch_size,
            max_queue_delay_us,
            queue_capacity: 1024,
            default_timeout_us: 5_000_000,
            llm: false,
            cache_blocks: 256,
            cache_block_tokens: 16,
            inputs: vec![
                TensorMetadata {
                    name: "INPUT0".to_owned(),
                    datatype: "FP64".to_owned(),
                    shape: vec![-1, 1],
                },
                TensorMetadata {
                    name: "INPUT1".to_owned(),
                    datatype: "FP64".to_owned(),
                    shape: vec![-1, 1],
                },
            ],
            outputs: vec![TensorMetadata {
                name: "OUTPUT0".to_owned(),
                datatype: "FP64".to_owned(),
                shape: vec![-1, 1],
            }],
        },
        directory: PathBuf::new(),
    }
}

fn request(batch: i64) -> InferRequest {
    let data = (0..batch).map(|value| json!(value)).collect::<Vec<_>>();
    InferRequest {
        id: "bench".to_owned(),
        parameters: BTreeMap::new(),
        inputs: vec![
            InferTensor {
                name: "INPUT0".to_owned(),
                shape: vec![batch, 1],
                datatype: "FP64".to_owned(),
                data: json!(data),
            },
            InferTensor {
                name: "INPUT1".to_owned(),
                shape: vec![batch, 1],
                datatype: "FP64".to_owned(),
                data: json!(data),
            },
        ],
        outputs: Vec::new(),
    }
}

/// Per-request admission cost: schema validation plus the batch-compatibility
/// signature the batcher groups on.
fn admission(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("admission");
    for batch in [1_i64, 8, 64] {
        let request = request(batch);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            format!("validate/batch-{batch}"),
            &request,
            |bencher, request| {
                bencher.iter(|| request.validate().unwrap());
            },
        );
        group.bench_with_input(
            format!("shape_signature/batch-{batch}"),
            &request,
            |bencher, request| {
                bencher.iter(|| request.shape_signature());
            },
        );
    }
    group.finish();
}

/// Per-batch routing cost: eligibility filtering plus affinity- and
/// load-ordered worker selection.
fn worker_selection(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("worker_selection");
    let key = model("adder", 16, 2_000).key;
    for workers in [2_usize, 8, 32] {
        let registry = WorkerRegistry::new(600_000);
        for number in 0..workers {
            registry
                .register(WorkerRegistration {
                    id: format!("worker-{number:03}"),
                    capacity: 4,
                    models: BTreeSet::from([key.clone()]),
                })
                .unwrap();
        }
        let excluded = BTreeSet::new();
        group.throughput(Throughput::Elements(1));
        group.bench_function(format!("reserve_release/workers-{workers}"), |bencher| {
            bencher.iter(|| {
                let selection = registry.reserve(&key, None, &excluded).unwrap();
                registry.release(&selection.id);
            });
        });

        // Affinity routing has to consult each candidate's cached prefix set,
        // so it is measured separately from the unaffiliated path.
        registry
            .heartbeat("worker-000", ["prefix-hot".to_owned()])
            .unwrap();
        group.bench_function(format!("reserve_affinity/workers-{workers}"), |bencher| {
            bencher.iter(|| {
                let selection = registry
                    .reserve(&key, Some("prefix-hot"), &excluded)
                    .unwrap();
                registry.release(&selection.id);
            });
        });
    }
    group.finish();
}

/// Per-LLM-request prefix-cache cost, split by how much of the prefix is
/// already resident.
fn prefix_cache(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("prefix_cache");
    let tokens = (0..256_u32).collect::<Vec<_>>();

    group.throughput(Throughput::Elements(1));
    group.bench_function("acquire_release/hit", |bencher| {
        let mut cache = PagedKvCache::new(1024, 16).unwrap();
        // Prime the exact prefix so every iteration takes the hit path.
        let primed = cache.acquire(&tokens).unwrap();
        cache.release(&primed).unwrap();
        bencher.iter(|| {
            let lease = cache.acquire(&tokens).unwrap();
            cache.release(&lease).unwrap();
        });
    });

    group.bench_function("acquire_release/cold", |bencher| {
        // A fresh cache per iteration keeps this on the allocate-and-evict path
        // rather than measuring a warm hit.
        bencher.iter_batched(
            || PagedKvCache::new(1024, 16).unwrap(),
            |mut cache| {
                let lease = cache.acquire(&tokens).unwrap();
                cache.release(&lease).unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("acquire_release/extend", |bencher| {
        let mut cache = PagedKvCache::new(1024, 16).unwrap();
        let base = cache.acquire(&tokens[..128]).unwrap();
        cache.release(&base).unwrap();
        // Longest-prefix reuse: shares the resident blocks, allocates the rest.
        bencher.iter(|| {
            let lease = cache.acquire(&tokens).unwrap();
            cache.release(&lease).unwrap();
        });
    });
    group.finish();
}

/// End-to-end in-process scheduling: admission, batching window, dispatch, and
/// response delivery, excluding HTTP and sockets.
fn scheduler_throughput(criterion: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let mut group = criterion.benchmark_group("scheduler");
    // A short window keeps the measurement dominated by scheduling work rather
    // than by waiting for a batch to fill.
    let scheduler = runtime.block_on(async {
        let repository =
            Arc::new(infermesh::repository::ModelRepository::open("model_repository").unwrap());
        let registry = Arc::new(WorkerRegistry::new(600_000));
        let engines = Arc::new(EngineDirectory::default());
        let metrics = Arc::new(Metrics::default());
        let models = repository
            .list()
            .into_iter()
            .map(|model| model.key)
            .collect::<BTreeSet<_>>();
        for number in 0..4 {
            let worker_id = format!("bench-{number}");
            registry
                .register(WorkerRegistration {
                    id: worker_id.clone(),
                    capacity: 4,
                    models: models.clone(),
                })
                .unwrap();
            engines.register(worker_id, Arc::new(DeterministicCpuEngine::default()));
        }
        let dispatcher = Arc::new(Dispatcher::new(
            Arc::clone(&registry),
            engines,
            Arc::clone(&metrics),
        ));
        Arc::new(Scheduler::new(repository, dispatcher, metrics))
    });

    for concurrency in [1_usize, 16, 64] {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_function(format!("infer/concurrency-{concurrency}"), |bencher| {
            bencher.to_async(&runtime).iter(|| {
                let scheduler = Arc::clone(&scheduler);
                async move {
                    let mut tasks = Vec::with_capacity(concurrency);
                    for _ in 0..concurrency {
                        let scheduler = Arc::clone(&scheduler);
                        tasks.push(tokio::spawn(async move {
                            scheduler.infer("adder", None, request(1)).await.unwrap()
                        }));
                    }
                    for task in tasks {
                        task.await.unwrap();
                    }
                }
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = admission, worker_selection, prefix_cache, scheduler_throughput
}
criterion_main!(benches);
