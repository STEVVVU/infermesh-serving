# Benchmark Methodology

The benchmark is a release-mode end-to-end control-plane workload:

```powershell
cargo run --release --bin infermesh-bench -- 2000 16
```

It:

1. Loads the checked-in repository.
2. Registers four deterministic CPU workers with four slots each.
3. Starts the actual bounded HTTP server on an ephemeral loopback port.
4. Synchronizes 16 client threads.
5. Sends 2,000 Triton V2 `adder` requests using one TCP connection each.
6. Records client-observed latency around connect, write, scheduling,
   execution, and response read.
7. Stops and joins the server.

Samples are sorted and nearest-rank p50/p95/p99 values are emitted. Throughput
counts successful responses only. Any non-200 response increments `errors`.

## Recorded run

The checked-in Windows x86-64 run completed 2,000/2,000 requests in 1.5432011
seconds:

- 1,296.01 successful requests/s
- p50 11.081 ms
- p95 20.213 ms
- p99 39.151 ms

The benchmark uses JSON tensors, loopback networking, deterministic scalar CPU
addition, max batch 16, and max queue delay 2 ms. It does not measure model
compute, ONNX Runtime, PyTorch, CUDA, GPU memory, PCIe transfer, or multi-host
networking. OS caches, scheduler load, TCP behavior, and CPU power state affect
results. The artifact is a reproducibility aid, not a service-level objective.
