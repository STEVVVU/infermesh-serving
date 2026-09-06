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

**`onnxruntime` stays an optional extra rather than a default dependency.** It
is the signature runtime for CPU inference serving, so the temptation is to
promote it — but the deciding factor is that nothing in the default path imports
it. `OnnxRuntimeAdapter` imports `onnxruntime` lazily inside `_session()`, only
once a model actually selects that backend, and raises a clear `AdapterError`
naming the extra if it is absent. Promoting it would add a ~200 MB native
dependency (plus `numpy`) to every install, including the `deterministic_cpu`
default and the Rust-only control-plane path, to satisfy an import that may
never happen. Keeping it optional also keeps `dependencies = []` honest for the
dependency-free CPU adapter. It is not unverified: the `onnx-runtime` CI job
installs `[onnx-test]` and runs a real `InferenceSession` end to end, so the
integration is proven on every push without being imposed on every user.

The optional Torch adapter loads `model.pt` using `torch.jit.load`,
`map_location="cpu"`, and `torch.inference_mode()`. It deliberately does not
place models on CUDA. Torch remains uninstalled and unverified in local and CI
evidence.

## Verify

These are the same commands CI runs, in the same order:

```powershell
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo test --benches                  # runs each criterion bench body once
$env:PYTHONPATH="$PWD/python"
python scripts/check_test_discovery.py python/tests 2
python -m unittest discover -s python/tests -v
python -m pip install "./python[onnx-test]"
python scripts/check_test_discovery.py python/tests_onnx 1
python -m unittest discover -s python/tests_onnx -v
```

`cargo test --benches` exists because `--all-targets` compiles the criterion
benches but never executes them, so a bench that panics could rot unnoticed.
Timings from that run are meaningless and deliberately discarded.

`check_test_discovery.py` guards a specific failure mode: `unittest discover`
exits **0** when it discovers **zero** tests, so a mistyped path or a missing
optional dependency yields a green "OK" that asserts nothing. The script
asserts a minimum count and separately rejects discovery *errors*, since an
unimportable module surfaces as a synthetic failing test that a plain count
would otherwise accept.

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
**twelve runs with zero errors**. A single loopback run is noisy, so the spread
is reported rather than one figure:

| Measurement | Median | Min | Max |
| --- | ---: | ---: | ---: |
| Throughput | 2,138.72 req/s | 1,840.27 | 2,803.30 |
| p50 latency | 6.282 ms | 4.378 ms | 7.132 ms |
| p95 latency | 15.311 ms | 13.007 ms | 18.210 ms |
| p99 latency | 23.772 ms | 20.061 ms | 43.721 ms |

The artifact is
[`benchmarks/windows-x86_64-2026-07-28-tokio-quiesced.json`](benchmarks/windows-x86_64-2026-07-28-tokio-quiesced.json).
These are development-machine CPU runs, not a capacity or GPU claim. The
measurement box is a power-managed laptop with unpinned clocks, so these are
envelopes rather than point values.

### Why there are three tokio artifacts for one day

This workload was measured three times on the same machine and got three
different answers, which is worth more than any of the individual numbers:

| Artifact | Throughput median | Conditions |
| --- | ---: | --- |
| [`-tokio`](benchmarks/windows-x86_64-2026-07-28-tokio.json) (7 runs) | 1,191 req/s | socket pressure uncontrolled |
| [`-remeasured`](benchmarks/windows-x86_64-2026-07-28-tokio-remeasured.json) (12 runs) | 1,204 req/s | socket pressure controlled, heavy background compiles running |
| [`-quiesced`](benchmarks/windows-x86_64-2026-07-28-tokio-quiesced.json) (12 runs) | 2,139 req/s | socket pressure controlled, machine otherwise idle |

Two distinct traps, both of which report **zero errors** while returning a bad
number:

- Windows `TIME_WAIT` accumulation. With one connection per request, throughput
  degrades *before* any connection is refused, so run spacing alone moves the
  result. Runs are now filtered on starting socket pressure.
- Unrelated CPU load. Re-running the byte-identical binary once background
  compiles finished raised throughput ~75%. Across the final twelve runs,
  throughput tracked machine utilisation closely: runs starting below 25% CPU
  returned 2,440–2,803 req/s, runs starting above 60% returned 1,840–1,941.

The earlier artifacts are kept rather than deleted, because the disagreement
between them is the honest result. The practical lesson is that a benchmark on
a busy developer machine can be off by ~1.5x while looking entirely plausible,
so the machine's load has to be recorded next to the number.

### What the async migration did to the numbers

The pre-async revision recorded a single run at 1,296.01 req/s, p50 11.081 ms,
p95 20.213 ms, p99 39.151 ms
([artifact](benchmarks/windows-x86_64-2026-07-17.json)). Honestly compared:

The single most important caveat is that **the baseline is one run whose machine
load was not recorded.** Since load alone moves this benchmark by ~1.5x, the
baseline cannot support a precise A/B claim in either direction. With that said,
every measurement moved favourably:

- **p50 improved**, 11.081 ms to a median of 6.282 ms (range 4.378–7.132).
  Requests no longer queue for one of 16 fixed connection threads.
- **p95 improved**, 20.213 ms to a median of 15.311 ms (range 13.007–18.210).
  The ranges do not overlap.
- **p99 improved**, 39.151 ms to a median of 23.772 ms — though this sample's
  max is 43.721 ms, above the baseline, so the tail is not uniformly better run
  to run.
- **Throughput is higher**, 1,296 to a median of 2,139 req/s, with the baseline
  below this sample's minimum of 1,840.

**No throughput or latency win is claimed as a measured result**, because a
one-run unlabelled baseline does not support one. What the migration is claimed
to buy is cancellation safety, per-request timeouts, and a concurrency model
that does not pin an OS thread per waiting request — properties that matter when
engines are actually slow, which is the real serving case. It is worth noting
that this benchmark's workload is trivial CPU arithmetic, so it flatters neither
design: async overhead is a larger share of total time here than it would be
with genuine model compute.

See [benchmark methodology](docs/benchmark.md) for the harness details,
including the Windows ephemeral-port exhaustion that degrades throughput and
eventually reports false connection errors on back-to-back runs.

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
  reload, Prometheus metrics, and graceful shutdown on tokio/axum/tower;
  measured a median 2,139 CPU requests/s (1,840–2,803 over 12 runs) with p95
  15.3 ms across 2,000 requests per run.
- Designed lease-based distributed worker routing with capacity-normalized
  least-load selection, KV-prefix affinity, heartbeat expiry, and
  phase-classified retries that replay only work proven not to have started.
- Implemented a refcounted paged KV cache with prefix block sharing and LRU
  eviction plus a versioned Rust/Python engine protocol; verified deterministic
  CPU and generated-model ONNX Runtime execution while keeping TorchScript an
  explicitly optional, unverified adapter.

## License

MIT
