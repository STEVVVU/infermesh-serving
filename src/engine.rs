use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::metrics::Metrics;
use crate::protocol::{InferRequest, InferResponse, InferTensor};
use crate::registry::WorkerRegistry;
use crate::repository::LoadedModel;
use crate::{Error, ErrorCode, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailurePhase {
    PreExecution,
    PostExecution,
}

#[derive(Debug)]
pub struct EngineFailure {
    pub phase: FailurePhase,
    pub message: String,
}

pub trait Engine: Send + Sync {
    fn execute(
        &self,
        model: &LoadedModel,
        requests: &[InferRequest],
    ) -> std::result::Result<Vec<InferResponse>, EngineFailure>;
}

const ENGINE_PROTOCOL: &str = "infermesh-engine/1";

struct ProcessState {
    child: Child,
    stdin: ChildStdin,
    responses: Receiver<String>,
}

impl Drop for ProcessState {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct ProcessEngine {
    state: Mutex<ProcessState>,
    timeout: Duration,
    next_request_id: AtomicUsize,
}

#[derive(Deserialize)]
struct WorkerResponse {
    protocol: String,
    request_id: String,
    status: String,
    #[serde(default)]
    responses: Vec<InferResponse>,
    #[serde(default)]
    error: String,
    #[serde(default)]
    execution_started: bool,
}

impl ProcessEngine {
    pub fn spawn(python: &str, backend: &str, timeout: Duration) -> Result<Self> {
        let python_path = std::env::current_dir()?.join("python");
        let mut child = Command::new(python)
            .args(["-m", "infermesh_worker.worker", "--backend", backend])
            .env("PYTHONPATH", python_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                Error::new(
                    ErrorCode::Unavailable,
                    format!("failed to start Python worker: {error}"),
                )
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::new(ErrorCode::Internal, "worker stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::new(ErrorCode::Internal, "worker stdout unavailable"))?;
        let (sender, responses) = mpsc::channel();
        thread::Builder::new()
            .name(format!("python-{backend}-responses"))
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else {
                        break;
                    };
                    if sender.send(line).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| {
                Error::new(
                    ErrorCode::Internal,
                    format!("failed to start worker response reader: {error}"),
                )
            })?;
        Ok(Self {
            state: Mutex::new(ProcessState {
                child,
                stdin,
                responses,
            }),
            timeout,
            next_request_id: AtomicUsize::new(1),
        })
    }
}

impl Engine for ProcessEngine {
    fn execute(
        &self,
        model: &LoadedModel,
        requests: &[InferRequest],
    ) -> std::result::Result<Vec<InferResponse>, EngineFailure> {
        let request_id = self
            .next_request_id
            .fetch_add(1, Ordering::Relaxed)
            .to_string();
        let envelope = json!({
            "protocol": ENGINE_PROTOCOL,
            "request_id": request_id,
            "model": {
                "name": model.key.name,
                "version": model.key.version,
                "directory": model.directory,
                "backend": model.config.backend,
            },
            "requests": requests,
        });
        let mut encoded = serde_json::to_vec(&envelope).map_err(|error| EngineFailure {
            phase: FailurePhase::PreExecution,
            message: error.to_string(),
        })?;
        encoded.push(b'\n');
        let mut state = self.state.lock().expect("process engine lock poisoned");
        if state
            .child
            .try_wait()
            .map_err(|error| EngineFailure {
                phase: FailurePhase::PreExecution,
                message: error.to_string(),
            })?
            .is_some()
        {
            return Err(EngineFailure {
                phase: FailurePhase::PreExecution,
                message: "Python worker exited before request dispatch".to_owned(),
            });
        }
        state
            .stdin
            .write_all(&encoded)
            .and_then(|()| state.stdin.flush())
            .map_err(|error| EngineFailure {
                phase: FailurePhase::PreExecution,
                message: format!("failed to send request to Python worker: {error}"),
            })?;
        let line = match state.responses.recv_timeout(self.timeout) {
            Ok(line) => line,
            Err(RecvTimeoutError::Timeout) => {
                let _ = state.child.kill();
                return Err(EngineFailure {
                    phase: FailurePhase::PostExecution,
                    message: "Python worker response timeout; process terminated".to_owned(),
                });
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(EngineFailure {
                    phase: FailurePhase::PostExecution,
                    message: "Python worker response stream closed".to_owned(),
                });
            }
        };
        let response: WorkerResponse =
            serde_json::from_str(&line).map_err(|error| EngineFailure {
                phase: FailurePhase::PostExecution,
                message: format!("invalid Python worker response: {error}"),
            })?;
        if response.protocol != ENGINE_PROTOCOL || response.request_id != request_id {
            return Err(EngineFailure {
                phase: FailurePhase::PostExecution,
                message: "Python worker protocol/request id mismatch".to_owned(),
            });
        }
        if response.status != "ok" {
            return Err(EngineFailure {
                phase: if response.execution_started {
                    FailurePhase::PostExecution
                } else {
                    FailurePhase::PreExecution
                },
                message: response.error,
            });
        }
        if response.responses.len() != requests.len() {
            return Err(EngineFailure {
                phase: FailurePhase::PostExecution,
                message: "Python worker response count mismatch".to_owned(),
            });
        }
        Ok(response.responses)
    }
}

#[derive(Debug, Default)]
pub struct DeterministicCpuEngine {
    delay_us: u64,
    pre_execution_failures: AtomicUsize,
}

impl DeterministicCpuEngine {
    #[must_use]
    pub fn with_delay(delay_us: u64) -> Self {
        Self {
            delay_us,
            pre_execution_failures: AtomicUsize::new(0),
        }
    }

    #[must_use]
    pub fn failing_before_execution(failures: usize) -> Self {
        Self {
            delay_us: 0,
            pre_execution_failures: AtomicUsize::new(failures),
        }
    }
}

impl Engine for DeterministicCpuEngine {
    fn execute(
        &self,
        model: &LoadedModel,
        requests: &[InferRequest],
    ) -> std::result::Result<Vec<InferResponse>, EngineFailure> {
        if self
            .pre_execution_failures
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
        {
            return Err(EngineFailure {
                phase: FailurePhase::PreExecution,
                message: "injected failure before execution".to_owned(),
            });
        }
        if self.delay_us > 0 {
            thread::sleep(Duration::from_micros(self.delay_us));
        }
        requests
            .iter()
            .map(|request| execute_cpu(model, request))
            .collect()
    }
}

fn execute_cpu(
    model: &LoadedModel,
    request: &InferRequest,
) -> std::result::Result<InferResponse, EngineFailure> {
    let outputs = if model.key.name == "adder" {
        if request.inputs.len() != 2 {
            return Err(post_failure("adder requires exactly two inputs"));
        }
        let left = numeric_values(&request.inputs[0].data)?;
        let right = numeric_values(&request.inputs[1].data)?;
        if left.len() != right.len() {
            return Err(post_failure("adder input lengths differ"));
        }
        let values = left
            .iter()
            .zip(right)
            .map(|(left, right)| Value::from(left + right))
            .collect();
        vec![InferTensor {
            name: model.config.outputs[0].name.clone(),
            shape: request.inputs[0].shape.clone(),
            datatype: "FP64".to_owned(),
            data: Value::Array(values),
        }]
    } else {
        let input = request
            .inputs
            .first()
            .ok_or_else(|| post_failure("identity backend requires an input"))?;
        vec![InferTensor {
            name: model
                .config
                .outputs
                .first()
                .map_or_else(|| input.name.clone(), |output| output.name.clone()),
            shape: input.shape.clone(),
            datatype: input.datatype.clone(),
            data: input.data.clone(),
        }]
    };
    Ok(InferResponse {
        model_name: model.key.name.clone(),
        model_version: model.key.version.clone(),
        id: request.id.clone(),
        outputs,
        parameters: BTreeMap::from([("backend".to_owned(), json!("deterministic_cpu"))]),
    })
}

fn numeric_values(value: &Value) -> std::result::Result<Vec<f64>, EngineFailure> {
    value
        .as_array()
        .ok_or_else(|| post_failure("numeric tensor data is not an array"))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| post_failure("tensor contains a non-numeric value"))
        })
        .collect()
}

fn post_failure(message: &str) -> EngineFailure {
    EngineFailure {
        phase: FailurePhase::PostExecution,
        message: message.to_owned(),
    }
}

#[derive(Default)]
pub struct EngineDirectory {
    engines: RwLock<BTreeMap<String, Arc<dyn Engine>>>,
}

impl EngineDirectory {
    pub fn register(&self, worker_id: impl Into<String>, engine: Arc<dyn Engine>) {
        self.engines
            .write()
            .expect("engine directory lock poisoned")
            .insert(worker_id.into(), engine);
    }

    pub fn remove(&self, worker_id: &str) {
        self.engines
            .write()
            .expect("engine directory lock poisoned")
            .remove(worker_id);
    }

    fn get(&self, worker_id: &str) -> Option<Arc<dyn Engine>> {
        self.engines
            .read()
            .expect("engine directory lock poisoned")
            .get(worker_id)
            .cloned()
    }
}

pub struct Dispatcher {
    registry: Arc<WorkerRegistry>,
    engines: Arc<EngineDirectory>,
    metrics: Arc<Metrics>,
}

pub struct DispatchResult {
    pub responses: Vec<InferResponse>,
    pub worker_id: String,
    pub retried: bool,
}

impl Dispatcher {
    #[must_use]
    pub fn new(
        registry: Arc<WorkerRegistry>,
        engines: Arc<EngineDirectory>,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            registry,
            engines,
            metrics,
        }
    }

    pub fn dispatch(
        &self,
        model: &LoadedModel,
        requests: &[InferRequest],
        prefix_key: Option<&str>,
    ) -> Result<DispatchResult> {
        let mut excluded = BTreeSet::new();
        let mut retried = false;
        for _ in 0..2 {
            let selection = self.registry.reserve(&model.key, prefix_key, &excluded)?;
            let Some(engine) = self.engines.get(&selection.id) else {
                self.registry.release(&selection.id);
                excluded.insert(selection.id);
                retried = true;
                self.metrics
                    .safe_retries_total
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            };
            match engine.execute(model, requests) {
                Ok(responses) => {
                    self.registry.release(&selection.id);
                    if let Some(prefix_key) = prefix_key {
                        self.registry
                            .note_cached_prefix(&selection.id, prefix_key.to_owned());
                    }
                    return Ok(DispatchResult {
                        responses,
                        worker_id: selection.id,
                        retried,
                    });
                }
                Err(failure) => {
                    self.registry.release(&selection.id);
                    self.metrics
                        .worker_failures_total
                        .fetch_add(1, Ordering::Relaxed);
                    if failure.phase == FailurePhase::PreExecution {
                        excluded.insert(selection.id);
                        retried = true;
                        self.metrics
                            .safe_retries_total
                            .fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    return Err(Error::new(
                        ErrorCode::Internal,
                        format!(
                            "worker {} failed after execution began: {}",
                            selection.id, failure.message
                        ),
                    ));
                }
            }
        }
        Err(Error::new(
            ErrorCode::Unavailable,
            format!(
                "all candidate workers failed before executing {}:{}",
                model.key.name, model.key.version
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    use serde_json::json;

    use super::{
        DeterministicCpuEngine, Dispatcher, Engine, EngineDirectory, EngineFailure, FailurePhase,
        ProcessEngine,
    };
    use crate::metrics::Metrics;
    use crate::protocol::{InferRequest, InferTensor, TensorMetadata};
    use crate::registry::{WorkerRegistration, WorkerRegistry};
    use crate::repository::{LoadedModel, ModelConfig, ModelKey};

    fn model() -> LoadedModel {
        LoadedModel {
            key: ModelKey {
                name: "adder".to_owned(),
                version: "1".to_owned(),
            },
            config: ModelConfig {
                name: "adder".to_owned(),
                platform: "cpu".to_owned(),
                backend: "deterministic_cpu".to_owned(),
                max_batch_size: 8,
                max_queue_delay_us: 100,
                queue_capacity: 8,
                default_timeout_us: 100_000,
                llm: false,
                cache_blocks: 8,
                cache_block_tokens: 4,
                inputs: Vec::new(),
                outputs: vec![TensorMetadata {
                    name: "OUTPUT0".to_owned(),
                    datatype: "FP64".to_owned(),
                    shape: vec![-1, 1],
                }],
            },
            directory: std::path::PathBuf::new(),
        }
    }

    fn request() -> InferRequest {
        InferRequest {
            id: "request".to_owned(),
            parameters: std::collections::BTreeMap::default(),
            inputs: vec![
                InferTensor {
                    name: "A".to_owned(),
                    shape: vec![1, 1],
                    datatype: "FP64".to_owned(),
                    data: json!([2.0]),
                },
                InferTensor {
                    name: "B".to_owned(),
                    shape: vec![1, 1],
                    datatype: "FP64".to_owned(),
                    data: json!([3.0]),
                },
            ],
            outputs: Vec::new(),
        }
    }

    #[test]
    fn retries_only_pre_execution_failure_on_another_worker() {
        let registry = Arc::new(WorkerRegistry::new(60_000));
        let engines = Arc::new(EngineDirectory::default());
        for id in ["a", "b"] {
            registry
                .register(WorkerRegistration {
                    id: id.to_owned(),
                    capacity: 1,
                    models: BTreeSet::from([model().key]),
                })
                .unwrap();
        }
        engines.register(
            "a",
            Arc::new(DeterministicCpuEngine::failing_before_execution(1)),
        );
        engines.register("b", Arc::new(DeterministicCpuEngine::default()));
        let dispatcher = Dispatcher::new(registry, engines, Arc::new(Metrics::default()));
        let result = dispatcher.dispatch(&model(), &[request()], None).unwrap();
        assert!(result.retried);
        assert_eq!(result.worker_id, "b");
        assert_eq!(result.responses[0].outputs[0].data, json!([5.0]));
    }

    #[test]
    fn python_cpu_process_implements_engine_protocol() {
        let engine =
            ProcessEngine::spawn("python", "cpu", std::time::Duration::from_secs(5)).unwrap();
        let responses = engine.execute(&model(), &[request()]).unwrap();
        assert_eq!(responses[0].outputs[0].data, json!([5.0]));
        assert_eq!(responses[0].parameters["backend"], json!("python_cpu"));
    }

    struct PostExecutionFailure;

    impl Engine for PostExecutionFailure {
        fn execute(
            &self,
            _model: &LoadedModel,
            _requests: &[InferRequest],
        ) -> std::result::Result<Vec<crate::protocol::InferResponse>, EngineFailure> {
            Err(EngineFailure {
                phase: FailurePhase::PostExecution,
                message: "ambiguous completion".to_owned(),
            })
        }
    }

    #[test]
    fn post_execution_failure_is_never_retried() {
        let registry = Arc::new(WorkerRegistry::new(60_000));
        let engines = Arc::new(EngineDirectory::default());
        for id in ["a", "b"] {
            registry
                .register(WorkerRegistration {
                    id: id.to_owned(),
                    capacity: 1,
                    models: BTreeSet::from([model().key]),
                })
                .unwrap();
        }
        engines.register("a", Arc::new(PostExecutionFailure));
        engines.register("b", Arc::new(DeterministicCpuEngine::default()));
        let metrics = Arc::new(Metrics::default());
        let dispatcher = Dispatcher::new(registry, engines, Arc::clone(&metrics));
        let error = dispatcher
            .dispatch(&model(), &[request()], None)
            .err()
            .unwrap();
        assert_eq!(error.code, crate::ErrorCode::Internal);
        assert_eq!(metrics.safe_retries_total.load(Ordering::Relaxed), 0);
    }
}
