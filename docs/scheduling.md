# Scheduling

## Dynamic batching

Each loaded model version owns a bounded synchronous queue. Admission uses
nonblocking `try_send`; a full queue returns resource exhaustion immediately.

The batcher:

1. Takes the oldest request and rejects it if its deadline elapsed.
2. Computes a signature from ordered input name, datatype, rank, and every
   dimension except the leading batch dimension.
3. Collects until maximum queue delay or maximum total batch size.
4. Defers incompatible requests to a FIFO backlog.
5. Rechecks deadlines immediately before and after execution.

Queue delay is measured from the first request's enqueue time, not from when
the batcher notices it. This prevents backlog time from extending the policy.
Requests that time out while an engine is running do not receive late output;
engine cancellation is not implemented.

## Worker routing

Eligible workers must:

- hold an unexpired heartbeat lease,
- advertise the exact model/version,
- have `in_flight < capacity`, and
- not be excluded after a prior pre-execution failure.

Candidates are ordered by:

1. exact KV-prefix cache affinity,
2. lowest normalized load,
3. lexical worker ID.

A reservation increments in-flight count before dispatch and releases it on
every response path.

## Retry boundary

InferMesh retries once on another worker only when the engine transport proves
execution did not begin. Examples are a missing engine mapping, exited Python
worker before write, or adapter model-load failure.

Timeout, disconnected response after dispatch, invalid response, and backend
execution errors are post-execution or ambiguous. They are never replayed
because inference handlers may have stateful or externally visible effects.

## Model reload

Reload rescans version directories and reconciles batchers. Existing versions
retain their batcher; new versions receive one; removed versions stop receiving
new traffic after their handle leaves the map. Workers must advertise support
for new versions before readiness becomes true.
