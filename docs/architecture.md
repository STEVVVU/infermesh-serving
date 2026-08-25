# Architecture

InferMesh separates request serving and scheduling from model execution.

```text
Triton V2 client
       |
       v
+------------------+   bounded connection queue
| Rust HTTP workers| ------------------------------+
+------------------+                               |
       |                                            v
       |                              +--------------------------+
       +-- metadata/health ---------->| model repository         |
       |                              | version routing + reload |
       |                              +--------------------------+
       v
+------------------+   compatible requests   +-------------------+
| per-model queue  | ----------------------> | dynamic batcher   |
| bounded admission|                         | deadline checks   |
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

One dedicated thread owns each model-version queue. This avoids cross-model
head-of-line blocking and makes batching policy local to the model config.
HTTP threads block on a one-shot result while queue capacity bounds accepted
work.

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

- HTTP uses 16 fixed workers and a bounded 256-connection queue.
- Every model queue is bounded independently.
- A model batcher is single-threaded; different models batch concurrently.
- Registry, engine directory, metrics, and cache accounting use scoped locks or
  atomics.
- Worker capacity limits concurrent dispatch. The deterministic demo registers
  two workers with four slots each.

## Lifecycle

Ctrl-C or `/v2/control/shutdown` stops admission, drains accepted connections,
joins HTTP workers, stops local heartbeats, and emits structured lifecycle
events. The shutdown endpoint is intended for local orchestration and is not
safe to expose without authentication.
