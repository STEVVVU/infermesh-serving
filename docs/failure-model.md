# Failure Model

## Covered failures

| Failure | Behavior |
| --- | --- |
| Model config malformed during reload | Reload fails before replacing current repository generation |
| Admission queue full | Immediate resource-exhausted response |
| Deadline expires in queue | Request removed without execution |
| Deadline expires during execution | Late output discarded; no retry |
| Worker heartbeat expires | Worker is excluded and can be removed deterministically |
| Worker fails before execution | One retry on a different eligible worker |
| Worker fails after/possibly after execution | Error returned without retry |
| Python worker exits before dispatch | Safe pre-execution failure |
| Python worker times out after dispatch | Process killed; treated as ambiguous post-execution failure |
| HTTP connection queue full | Immediate HTTP 503 |

## Worker loss

Registry state is in-memory and heartbeat leases use control-plane wall clock.
A process pause longer than the lease can temporarily exclude a healthy worker.
Re-registration is idempotent by worker ID. Reservations are released when the
dispatcher observes a return path; a control-plane crash loses all reservations
and registry state.

There is no distributed consensus, durable worker registry, or leader
election. "Distributed worker" in this revision means independently identified
capacity/lease/cache domains routed by one control plane.

## Delivery semantics

Inference dispatch is at-most-once after the execution boundary and may return
an ambiguous failure. Only work proven not to have started is retried. This
trades availability for protection against duplicate stateful execution.

## Shutdown

Shutdown stops acceptance and waits for admitted HTTP handlers. Model batcher
threads end when their senders are dropped at process teardown. There is no
configurable maximum drain deadline or forced cancellation.

## Uncovered failures

- Control-plane crash recovery or durable request queues
- Network partitions and remote worker RPC
- GPU reset, CUDA OOM, NCCL failure, or device health
- Framework allocator fragmentation
- Byzantine workers or compromised model artifacts
