# Benchmark Methodology

There are two benchmarks and they answer different questions. Neither
substitutes for the other.

## 1. End-to-end HTTP load (`infermesh-bench`)

A release-mode workload over real loopback TCP:

```powershell
cargo run --release --bin infermesh-bench -- 2000 16
```

It:

1. Loads the checked-in repository.
2. Registers four deterministic CPU workers with four slots each.
3. Starts the actual axum server on an ephemeral loopback port, on its own
   Tokio runtime.
4. Synchronizes 16 client tasks on a second, separate runtime.
5. Sends 2,000 Triton V2 `adder` requests using one TCP connection each.
6. Records client-observed latency around connect, write, scheduling,
   execution, and response read.
7. Signals a graceful drain and joins the server.

Samples are sorted and nearest-rank p50/p95/p99 values are emitted. Throughput
counts successful responses only. Any non-200 response increments `errors`.

### Why the runtimes are split

Running the load generator on the server's runtime makes client tasks compete
with the server's worker threads for the same cores. That shows up as server
tail latency when it is really client scheduling delay. The previous
thread-based harness gave clients their own OS threads and so did not have this
problem; splitting the runtimes preserves that property.

Latency samples are also merged once per client rather than once per request,
so the shared lock is no longer inside the measured path.

### Recorded run

Measured on 2026-07-28, Windows x86-64, 14 logical CPUs, Rust 1.96.0, across
**seven** runs with **zero errors** in total. A single loopback run is noisy, so
the distribution is reported rather than one number:

| Measurement | Median | Min | Max |
| --- | ---: | ---: | ---: |
| Throughput (req/s) | 1,191.05 | 1,025.25 | 1,720.21 |
| p50 latency | 6.834 ms | 5.928 ms | 8.711 ms |
| p95 latency | 33.316 ms | 23.782 ms | 57.932 ms |
| p99 latency | 103.562 ms | 63.075 ms | 141.834 ms |

The artifact is
[`benchmarks/windows-x86_64-2026-07-28-tokio.json`](../benchmarks/windows-x86_64-2026-07-28-tokio.json).

### Comparison with the pre-async revision

The previous artifact
([`windows-x86_64-2026-07-17.json`](../benchmarks/windows-x86_64-2026-07-17.json))
recorded a single run at 1,296.01 req/s, p50 11.081 ms, p95 20.213 ms, p99
39.151 ms.

Read that comparison carefully, because it is not a clean A/B:

- **p50 improved substantially** (11.081 ms to 6.834 ms). Requests no longer
  wait for one of 16 fixed connection threads.
- **Median throughput is slightly lower** (1,296 to 1,191 req/s) but the old
  figure is a single sample, and the new range (1,025 to 1,720) contains it.
  Treating a 1-sample number as a baseline overstates the precision available.
- **Tail latency is worse** (p99 39 ms to 104 ms). This is the real regression.
  The workload is deliberately trivial CPU arithmetic, so the async machinery is
  a larger share of total time than it would be with genuine model compute, and
  the concurrency-limit-plus-shed stack admits more work concurrently than 16
  threads did, which spreads latency out rather than queueing at accept.

Async is not a throughput win on a workload this cheap and is not claimed as
one. It buys cancellation safety, per-request timeouts, and a concurrency model
that does not pin an OS thread per waiting request. Those matter when engines
are slow, which is the real serving case.

### A harness artifact worth knowing

Running this benchmark back-to-back exhausts the Windows ephemeral port range:
with one connection per request, `TIME_WAIT` entries plateau near 32,000 and
connections start being refused. Those errors are a property of the load
generator, not the server. Space runs apart, or expect false failures.

## 2. Scheduling microbenchmarks (`cargo bench`)

Criterion benchmarks over the in-process hot paths, with no sockets involved:

```powershell
cargo bench
```

- `admission` — per-request schema validation and shape-signature computation
  at batch sizes 1/8/64.
- `worker_selection` — reserve/release against 2/8/32 registered workers, with
  and without prefix-cache affinity.
- `prefix_cache` — paged KV-cache acquire/release on the hit, cold, and
  longest-prefix-extend paths.
- `scheduler` — end-to-end in-process scheduling at concurrency 1/16/64.

These isolate control-plane cost from network and OS effects, which the HTTP
benchmark cannot.

## What neither benchmark measures

Model compute, ONNX Runtime, PyTorch, CUDA, GPU memory, PCIe transfer, or
multi-host networking. OS caches, scheduler load, TCP behavior, and CPU power
state affect results. These artifacts are reproducibility aids, not
service-level objectives.
