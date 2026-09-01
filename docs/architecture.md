# Architecture

InferMesh separates request serving and scheduling from model execution. The
HTTP surface and the scheduler are asynchronous (Tokio tasks); model execution
is synchronous and runs on the blocking pool.

```text
Triton V2 client
       |
       v
+------------------+   tower: trace / load shed / concurrency limit / timeout
| axum HTTP router | ------------------------------+
+------------------+                               |
       |                                            v
       |                              +--------------------------+
       +-- metadata/health ---------->| model repository         |
       |                              | version routing + reload |
       |                              +--------------------------+
       v
+------------------+   compatible requests   +-------------------+
| per-model queue  | ----------------------> | dynamic batcher   |
| bounded admission|                         | task, deadlines   |
+------------------+                         +---------+---------+
                                                       |
                      +--------------------------------+
                      v
            +--------------------+      +----------------------+
            | worker registry    |<---->| paged prefix cache   |
            | lease/load/affinity|      | refs/LRU/accounting  |
            +---------+----------+      +----------------------+
                      |
             +--------+---------+
             |                  |
             v                  v
      Rust CPU engine    Python engine process
                          CPU / ONNX / Torch
```

## Control plane

The filesystem repository is the desired model state. Each
`model/version/model.json` defines tensor metadata, backend, batching bounds,
deadline default, admission capacity, and optional LLM cache size. Reload
builds a complete replacement map before publishing it, so malformed state
does not partially replace a working generation.

One dedicated Tokio task owns each model-version queue. This avoids cross-model
head-of-line blocking and makes batching policy local to the model config. The
HTTP handler awaits a `oneshot` result while queue capacity bounds accepted
work.

Because a waiting request is a task rather than a parked OS thread, a caller
that goes away is observable: dropping the `oneshot::Receiver` closes it, and
the batcher discards such requests before they consume worker capacity
(`infermesh_client_cancelled_total`).

## Worker plane

Workers register model capabilities and slot capacity. Heartbeats renew a
lease and report cached prefix keys. Routing first prefers cache affinity, then
compares normalized load (`in_flight / capacity`), then worker ID for
deterministic ties.

The engine directory maps registry IDs to execution transports. This revision
ships in-process Rust CPU engines and persistent Python subprocess engines.
The registry/routing model is network-ready, but remote worker RPC and a
replicated registry are not implemented.

## Concurrency

- HTTP admits at most 256 concurrent *requests*, enforced by a `tower`
  concurrency limit with load shedding. Under tokio, connections are cheap, so
  the quantity worth bounding is request work rather than sockets. This replaces
  the previous fixed pool of 16 connection threads plus a 256-slot connection
  queue; an overloaded server still answers 503.
- A 30-second `tower-http` timeout bounds any single request. Per-request
  deadlines still come from `timeout_us` or the model config; this only catches
  pathological cases such as a stalled body.
- Every model queue is bounded independently.
- A model batcher is a single task, so batching order within a model stays
  serial while different models batch concurrently.
- Engines are synchronous and run on the blocking pool via `spawn_blocking`.
  The deterministic engine computes inline and the process engine does blocking
  subprocess stdio, so neither may occupy an async worker thread.
- Registry, engine directory, metrics, and cache accounting use scoped
  `std::sync` locks or atomics. These are held only across non-await
  computation, so a blocking mutex is correct here and cheaper than an async one.
- Worker capacity limits concurrent dispatch. The deterministic demo registers
  two workers with four slots each.

## Lifecycle

Ctrl-C and `/v2/control/shutdown` both signal one `tokio::sync::Notify`, which
drives axum's graceful shutdown: the listener stops accepting, admitted requests
run to completion, local heartbeats stop, and lifecycle events are emitted.
`notify_one` rather than `notify_waiters` is used deliberately, so a shutdown
requested before the server reaches its await point is not lost.

New connections are refused once a drain begins. That is intended behaviour, not
a defect — it is what lets in-flight work finish without admitting more.

The shutdown endpoint is intended for local orchestration and is not safe to
expose without authentication.
