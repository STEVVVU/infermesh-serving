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
**twelve** runs with **zero errors** in total. A single loopback run is noisy,
so the distribution is reported rather than one number:

| Measurement | Median | Min | Max |
| --- | ---: | ---: | ---: |
| Throughput (req/s) | 2,138.72 | 1,840.27 | 2,803.30 |
| p50 latency | 6.282 ms | 4.378 ms | 7.132 ms |
| p95 latency | 15.311 ms | 13.007 ms | 18.210 ms |
| p99 latency | 23.772 ms | 20.061 ms | 43.721 ms |

The artifact is
[`benchmarks/windows-x86_64-2026-07-28-tokio-quiesced.json`](../benchmarks/windows-x86_64-2026-07-28-tokio-quiesced.json).
The measurement box is a power-managed laptop with unpinned clocks, so these are
envelopes, not point values.

### Three measurements, three answers

This workload was measured three times on the same machine on the same date. All
three artifacts are retained, because their disagreement is more instructive
than any single number. They are distinguished by filename suffix rather than by
date:

| Artifact | Runs | Throughput median | Conditions |
| --- | ---: | ---: | --- |
| [`-tokio`](../benchmarks/windows-x86_64-2026-07-28-tokio.json) | 7 | 1,191 req/s | socket pressure uncontrolled |
| [`-remeasured`](../benchmarks/windows-x86_64-2026-07-28-tokio-remeasured.json) | 12 | 1,204 req/s | socket pressure controlled; heavy background compiles running |
| [`-quiesced`](../benchmarks/windows-x86_64-2026-07-28-tokio-quiesced.json) | 12 | 2,139 req/s | socket pressure controlled; machine otherwise idle |

The first re-measurement corrected a methodology gap: the original run's
envelopes were too narrow because socket pressure was not controlled for, so its
p50 (6.834 ms) and p99 (103.562 ms) did not reproduce. But that re-measurement
was itself taken on a loaded machine, and re-running the byte-identical binary
once the load cleared raised throughput by roughly 75%.

Both failures share a shape: **a run can be wrong while reporting zero errors.**
Two independent causes, detailed below and in the artifacts:

1. Windows `TIME_WAIT` accumulation degrades throughput before any connection is
   refused.
2. Unrelated CPU load on the measurement box dominates everything else.

Across the final twelve runs, throughput tracked whole-machine utilisation
sampled at run start: runs beginning below 25% CPU returned 2,440–2,803 req/s,
while runs beginning above 60% returned 1,840–1,941. Record the machine's load
next to the number, or the artifact cannot be interpreted later.

### Comparison with the pre-async revision

The previous artifact
([`windows-x86_64-2026-07-17.json`](../benchmarks/windows-x86_64-2026-07-17.json))
recorded a single run at 1,296.01 req/s, p50 11.081 ms, p95 20.213 ms, p99
39.151 ms.

Read that comparison carefully, because it is not a clean A/B. That baseline is
**a single run whose machine load was not recorded**, and load alone moves this
benchmark by roughly 1.5x — so it cannot support a precise claim in either
direction. Every measurement did move favourably:

- **p50 improved** (11.081 ms to a median of 6.282 ms). Requests no longer wait
  for one of 16 fixed connection threads.
- **p95 improved** (20.213 ms to a median of 15.311 ms), with non-overlapping
  ranges.
- **p99 improved** (39.151 ms to a median of 23.772 ms), though this sample's max
  of 43.721 ms exceeds the baseline, so the tail is not uniformly better run to
  run.
- **Throughput is higher** (1,296 to a median of 2,139 req/s), with the baseline
  below this sample's minimum of 1,840.

No throughput or latency win is claimed as a measured result, because a one-run
unlabelled baseline does not support one. What async buys is cancellation safety,
per-request timeouts, and a concurrency model that does not pin an OS thread per
waiting request. Those matter when engines are slow, which is the real serving
case. Note also that this workload is deliberately trivial CPU arithmetic, so it
flatters neither design: async machinery is a larger share of total time here
than it would be with genuine model compute.

### A harness artifact worth knowing

Running this benchmark back-to-back exhausts the Windows ephemeral port range:
with one connection per request, `TIME_WAIT` entries plateau near 32,000 and
connections start being refused. Those errors are a property of the load
generator, not the server. Space runs apart, or expect false failures.

**The dangerous part is what happens before the errors.** Throughput degrades
*silently* as `TIME_WAIT` accumulates, so a port-starved run reports a bad
number rather than failing. Measured during the 2026-07-28 re-runs:

| `TIME_WAIT` at run start | Throughput | p99 | Errors |
| ---: | ---: | ---: | ---: |
| under ~5,000 | 1,840–2,803 req/s | 20–44 ms | 0 |
| ~10,000 and above | 245–400 req/s | 350–897 ms | 0–17 |

So a run with **zero errors is not automatically a valid run**. Two further
traps found: 20 s of spacing was not enough to drain on this machine
(`TIME_WAIT` kept climbing across a 180 s pause, then aged out in bulk), and the
low-throughput runs mostly reported no errors at all. Check the starting socket
count — `netstat -an | grep -c TIME_WAIT` — and discard runs that begin under
pressure.

Then check the machine is idle, which turned out to matter more. See "Three
measurements, three answers" above: controlling socket pressure alone still left
a ~75% error, because unrelated compile jobs were competing for the same cores.

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
