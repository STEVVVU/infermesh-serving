# Threat Model and Limits

InferMesh 0.1 is a local systems project, not an internet-facing deployment.

## Trusted boundary

The operator, model repository, model files, Python environment, and registered
workers are trusted. Model adapters execute native framework code and may load
arbitrary serialized artifacts. TorchScript and ONNX files must come from a
trusted supply chain.

## Missing production controls

- No authentication, authorization, TLS, tenant identity, or per-tenant quota
- No sandbox around Python workers or model artifacts
- No request-content redaction in application-level errors
- No remote-worker transport authentication
- No durable registry, audit log, or configuration signature
- No cgroup/job-object CPU or memory limit
- No decompression because compressed/chunked bodies are rejected

The shutdown and repository reload endpoints are administrative and
unauthenticated. The server binds to loopback by default; exposing it on another
interface requires a trusted reverse proxy and access control.

## Resource limits

HTTP headers, bodies, tensor element counts, model queue lengths, connection
queue length, worker capacity, and KV blocks are bounded. However, JSON parsing
still allocates proportional to accepted body size, snapshots of request
values are cloned into batches, and Python framework memory is not controlled.

The concurrency limit bounds in-flight request work and sheds beyond it, and a
server-wide timeout bounds any single request. Connections themselves are no
longer the scarce resource they were under a fixed thread pool, but that also
means a client can hold many idle connections open cheaply. There is no
per-connection limit, per-IP quota, or fairness scheduler.

Bodies are capped by `DefaultBodyLimit` and buffered whole before deserialization,
so there is no streaming body parser. HTTP/2 is not enabled.

## GPU boundary

No CUDA API is called. The ONNX adapter explicitly requests
`CPUExecutionProvider`; the Torch adapter maps models to CPU. KV-cache blocks
are logical accounting only. GPU execution, kernels, memory throughput, and
latency are neither implemented nor measured.
