# InferMesh Serving

InferMesh is a production-shaped AI inference serving platform with a Rust
HTTP/control plane and Python model-worker adapters. It implements a useful
subset of the Triton V2 HTTP API, per-model dynamic batching, bounded
admission, lease-based worker routing, and a paged prefix KV cache.

The checked-in workload is deterministic CPU inference. **No GPU execution or
GPU performance is claimed.**

## Implemented system behavior

- Triton V2 live/ready, server metadata, model metadata/ready, versioned and
  latest-version inference routes
- Per-model bounded queues with maximum batch size, queue delay, compatible
  non-batch shapes, and request deadlines
- Filesystem model repository with numeric latest-version routing and atomic
  reload
- Capacity-aware worker registration, heartbeats, lease expiry, normalized
  least-load routing, and prefix-cache affinity
- Retry on explicit pre-execution failure only; ambiguous post-execution
  failures are returned without replay
- Refcounted paged KV-cache blocks, longest-prefix reuse, active-lease pinning,
  LRU eviction, and LLM scheduler integration
- Prometheus counters/histograms, structured JSON logs, health/readiness,
  bounded HTTP connections, and graceful drain on Ctrl-C or control endpoint
- Rust deterministic CPU engine, persistent Python engine protocol, real ONNX
  Runtime CPU adapter, and optional CPU TorchScript adapter

## Run locally

Requirements are Rust 1.89+ and Python 3.11+. Build and start the default Rust
CPU runtime:

```powershell
cargo build --release
cargo run --release --bin infermesh-server -- serve model_repository 127.0.0.1:8000
```

Query health and model metadata:

```powershell
curl.exe http://127.0.0.1:8000/v2/health/ready
curl.exe http://127.0.0.1:8000/v2/models/adder
```

Submit an inference request:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v2/models/adder/infer `
  -H "Content-Type: application/json" `
  -d '{"id":"demo","inputs":[{"name":"INPUT0","shape":[1,1],"datatype":"FP64","data":[2.0]},{"name":"INPUT1","shape":[1,1],"datatype":"FP64","data":[3.0]}]}'
```

Set `INFERMESH_ENGINE=python-cpu` to run each local worker through the
persistent Rust-to-Python engine protocol:

```powershell
$env:INFERMESH_ENGINE="python-cpu"
cargo run --release --bin infermesh-server -- serve
```

## Triton V2 subset

| Method | Path | Behavior |
| --- | --- | --- |
| `GET` | `/v2` | Server name, version, extensions |
| `GET` | `/v2/health/live` | Process liveness |
| `GET` | `/v2/health/ready` | Repository plus worker readiness |
| `GET` | `/v2/models/{model}` | Model metadata and versions |
| `GET` | `/v2/models/{model}/ready` | Model worker readiness |
| `POST` | `/v2/models/{model}/infer` | Latest-version inference |
| `POST` | `/v2/models/{model}/versions/{version}/infer` | Explicit version |
| `POST` | `/v2/repository/reload` | Rescan model repository |
| `GET` | `/metrics` | Prometheus text exposition |

JSON tensor payloads support `FP32`, `FP64`, `INT32`, and `INT64`. Binary tensor
extensions, shared memory, gRPC, sequence batching, decoupled responses, and
model unload are not implemented.

## Python adapters

The dependency-free CPU adapter is covered by unit, subprocess, and
Rust-to-Python integration tests.

The ONNX Runtime adapter is tested end to end: a dedicated integration test
generates a deterministic opset-13 Add model, validates it with ONNX, loads it
through `onnxruntime.InferenceSession` using `CPUExecutionProvider`, executes
named inputs/output, and asserts output values, shape, and datatype. Install
runtime dependencies with:

```powershell
python -m pip install -e "./python[onnx]"
```

Place `model.onnx` beside the version's `model.json`, set its backend to
`onnxruntime`, and launch with `INFERMESH_ENGINE=python-onnx`.

The optional Torch adapter loads `model.pt` using `torch.jit.load`,
`map_location="cpu"`, and `torch.inference_mode()`. It deliberately does not
place models on CUDA. Torch remains uninstalled and unverified in local and CI
evidence.

## Verify

```powershell
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
$env:PYTHONPATH="$PWD/python"
python -m unittest discover -s python/tests -v
python -m pip install "./python[onnx-test]"
python -m unittest discover -s python/tests_onnx -v
```

The deterministic suite covers compatible and incompatible batching,
deadlines, saturated admission, repository reload, capacity and affinity
routing, lease expiry, pre/post-execution retry safety, worker-process
protocol, cache refcounts/eviction/reuse, and a real TCP API lifecycle.
The dedicated Ubuntu CI job installs ONNX test dependencies and executes the
generated-model integration separately from dependency-free worker tests.

Local ONNX verification on July 17, 2026 used CPython 3.11.9, NumPy 2.4.6,
ONNX 1.22.0, and ONNX Runtime 1.27.0. The generated Add-model test passed in
0.243 seconds and verified named inputs/output, `[2, 2]` shape, `FP32`, values
`[11.0, 22.0, 33.0, 44.0]`, and `CPUExecutionProvider`. Torch was not
installed or executed.

## Reproducible benchmark

```powershell
cargo run --release --bin infermesh-bench -- 2000 16
```

This starts an ephemeral local server and sends 2,000 Triton V2 HTTP requests
from 16 client threads, using one TCP connection per request. On July 17, 2026,
Windows x86-64 with Rust 1.96.0 produced:

| Measurement | Result |
| --- | ---: |
| Successful / errors | 2,000 / 0 |
| Throughput | 1,296.01 requests/s |
| p50 latency | 11.081 ms |
| p95 latency | 20.213 ms |
| p99 latency | 39.151 ms |

The exact artifact is
[`benchmarks/windows-x86_64-2026-07-17.json`](benchmarks/windows-x86_64-2026-07-17.json).
This is one development-machine CPU run, not a capacity or GPU claim.

## Design and limits

- [Architecture](docs/architecture.md)
- [Triton and engine protocols](docs/protocol.md)
- [Batching and worker scheduling](docs/scheduling.md)
- [Paged KV-cache accounting](docs/kv-cache.md)
- [Failure model](docs/failure-model.md)
- [Benchmark methodology](docs/benchmark.md)
- [Threat model and limits](docs/threat-model.md)

## Resume bullets

- Built a Rust inference control plane implementing Triton V2 HTTP endpoints,
  shape-aware dynamic batching, bounded admission/deadlines, model-version
  reload, Prometheus metrics, and graceful shutdown; measured 1,296 CPU
  requests/s with p95 20.2 ms across 2,000 requests.
- Designed lease-based distributed worker routing with capacity-normalized
  least-load selection, KV-prefix affinity, heartbeat expiry, and
  phase-classified retries that replay only work proven not to have started.
- Implemented a refcounted paged KV cache with prefix block sharing and LRU
  eviction plus a versioned Rust/Python engine protocol; verified deterministic
  CPU and generated-model ONNX Runtime execution while keeping TorchScript an
  explicitly optional, unverified adapter.

## License

MIT
