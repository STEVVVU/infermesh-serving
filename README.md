# InferMesh Serving

InferMesh is a production-shaped AI inference serving platform with an async
Rust HTTP/control plane and Python model-worker adapters. It implements a useful
subset of the Triton V2 HTTP API, per-model dynamic batching, bounded
admission, lease-based worker routing, and a paged prefix KV cache.

The checked-in workload is deterministic CPU inference. **No GPU execution or
GPU performance is claimed.**

## Stack

| Concern | Crate | Why |
| --- | --- | --- |
| Async runtime | `tokio` 1.53 | The control plane is I/O bound on client sockets and worker engines. A request is a task, not a pinned OS thread. |
| HTTP surface | `axum` 0.8 | Routing and extraction for the Triton V2 paths. Replaces ~220 lines of hand-written HTTP/1.1 parsing. |
| Middleware | `tower` 0.5, `tower-http` 0.7 | Admission (concurrency limit + load shed), request timeouts, and request tracing belong in the service stack, not handlers. |
| Logging | `tracing` 0.1, `tracing-subscriber` 0.3 | Structured, level-aware, span-correlated events with `RUST_LOG` filtering and JSON or text output. |
| Metrics | `prometheus-client` 0.25 | Typed OpenMetrics families instead of hand-formatted exposition text. |
| Errors | `thiserror` 2, `anyhow` 1 | Derived library errors; contextual errors at binary boundaries. |
| CLI | `clap` 4.6 | Declarative flags for bind address, repository, engine, worker sizing, and log format. |
| Benchmarks | `criterion` 0.8 | Statistical measurement of the scheduling hot paths. |

The crate keeps `unsafe_code = "forbid"` and clippy `pedantic`. MSRV is
unchanged at **1.89** — `criterion` 0.8 needs 1.86 and is the highest-MSRV
crate in the resolved tree, so nothing forces a raise.

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
- Prometheus counters/histograms, structured `tracing` logs, health/readiness,
  bounded request concurrency with load shedding, and graceful drain on Ctrl-C
  or control endpoint
- Cancellation safety: a client that disconnects or times out has its queued
  request dropped before it consumes worker capacity
- Rust deterministic CPU engine, persistent Python engine protocol, real ONNX
  Runtime CPU adapter, and optional CPU TorchScript adapter

## Run locally

Requirements are Rust 1.89+ and Python 3.11+. Build and start the default Rust
CPU runtime:

```powershell
cargo build --release
cargo run --release --bin infermesh-server -- serve model_repository 127.0.0.1:8000
```

The server has a full CLI; `--help` lists everything:

```powershell
cargo run --release --bin infermesh-server -- serve --help
```

Useful flags are `--engine` (`rust-cpu`, `python-cpu`, `python-onnx`,
`python-torch`), `--workers` / `--worker-capacity`, `--lease-ms`, and
`--log-format` (`json` or `text`). Log verbosity honours `RUST_LOG`.

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

Run each local worker through the persistent Rust-to-Python engine protocol
with either the flag or the environment variable:

```powershell
cargo run --release --bin infermesh-server -- serve --engine python-cpu
# or, as before:
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

Wire compatibility was preserved deliberately through the axum migration: every
route keeps its path, method, status codes, and JSON body shape, including the
`{"error": "..."}` error envelope and a 400 (not a framework 422) for
unparseable bodies. This is covered by wire-level integration tests that speak
raw HTTP/1.1 over a TCP socket rather than calling the router in-process.

One intentional exposition difference: `prometheus-client` follows OpenMetrics,
so a counter's `# TYPE` line names the metric family without the `_total`
suffix while the sample line keeps it. Series names themselves are unchanged.

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
cargo run --release --bin infermesh-bench -- 2000 16   # end-to-end HTTP
cargo bench                                            # scheduling hot paths
```

The HTTP benchmark starts an ephemeral local server and sends 2,000 Triton V2
requests from 16 client tasks, one TCP connection per request, with the load
generator on a runtime separate from the server's.

Measured on July 28, 2026, Windows x86-64, 14 logical CPUs, Rust 1.96.0, over
**seven runs with zero errors**. A single loopback run is noisy, so the spread
is reported rather than one figure:

| Measurement | Median | Min | Max |
| --- | ---: | ---: | ---: |
| Throughput | 1,191.05 req/s | 1,025.25 | 1,720.21 |
| p50 latency | 6.834 ms | 5.928 ms | 8.711 ms |
| p95 latency | 33.316 ms | 23.782 ms | 57.932 ms |
| p99 latency | 103.562 ms | 63.075 ms | 141.834 ms |

The artifact is
[`benchmarks/windows-x86_64-2026-07-28-tokio.json`](benchmarks/windows-x86_64-2026-07-28-tokio.json).
These are development-machine CPU runs, not a capacity or GPU claim.

### What the async migration did to the numbers

The pre-async revision recorded a single run at 1,296.01 req/s, p50 11.081 ms,
p95 20.213 ms, p99 39.151 ms
([artifact](benchmarks/windows-x86_64-2026-07-17.json)). Honestly compared:

- **p50 improved substantially**, 11.081 ms to 6.834 ms. Requests no longer
  queue for one of 16 fixed connection threads.
- **Median throughput is slightly lower**, 1,296 to 1,191 req/s — but the old
  number is a single sample and the new range (1,025–1,720) contains it. It
  would overstate the data to call this a measured regression.
- **Tail latency got worse**, p99 39 ms to 104 ms. This is a real regression.
  The benchmark workload is trivial CPU arithmetic, so async overhead is a
  larger share of total time than it would be with real model compute, and the
  concurrency-limit-plus-shed stack admits more work concurrently than 16
  threads did, spreading latency instead of queueing at accept.

**Async is not claimed as a throughput win here, because on a workload this
cheap it is not one.** What it buys is cancellation safety, per-request
timeouts, and a concurrency model that does not pin an OS thread per waiting
request — properties that matter when engines are actually slow, which is the
real serving case.

See [benchmark methodology](docs/benchmark.md) for the harness details,
including the Windows ephemeral-port exhaustion that makes back-to-back runs
report false connection errors.

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
