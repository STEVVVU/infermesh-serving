# Protocols

## Triton V2 HTTP subset

InferMesh preserves Triton-style model paths, JSON fields, tensor datatypes,
health semantics, and model metadata. Inference accepts:

```json
{
  "id": "request-42",
  "parameters": {
    "timeout_us": 500000,
    "prefix_tokens": [10, 11, 12]
  },
  "inputs": [
    {
      "name": "INPUT0",
      "shape": [1, 4],
      "datatype": "FP32",
      "data": [1.0, 2.0, 3.0, 4.0]
    }
  ],
  "outputs": [{"name": "OUTPUT0"}]
}
```

All dimensions must be positive in requests. Data is a flat JSON array whose
length must equal the shape product. The first dimension contributes to total
batch size; remaining dimensions form the compatibility signature.

The implementation limits request bodies to 4 MiB, tensors to 1,000,000
elements, and requests to 64 inputs. Header limits and chunked-transfer handling
are now hyper's rather than hand-written; chunked bodies are accepted by the
transport and still bounded by the 4 MiB body limit, where the previous
hand-rolled parser rejected them outright.

Errors are returned as `{"error": "..."}` with the status implied by the
control-plane error code (400 invalid argument, 404 not found, 405 method not
allowed, 408 deadline exceeded, 429 resource exhausted, 503 unavailable, 500
internal). An unparseable body is a 400, not a framework-default 422.

Every error status carries that JSON body, including the two the framework would
otherwise answer on its own: an unknown path is a 404 with
`{"error": "route not found"}`, and a known path addressed with an unsupported
method is a 405 with `{"error": "method not allowed for route"}` plus an `Allow`
header naming the methods that path does accept. A client may therefore read
`body["error"]` on any non-2xx response without first special-casing the status.

## Engine protocol

Rust and Python communicate over newline-delimited JSON on stdin/stdout. Each
line is limited to 8 MiB and carries `"protocol": "infermesh-engine/1"`.

Request envelope:

```json
{
  "protocol": "infermesh-engine/1",
  "request_id": "17",
  "model": {
    "name": "adder",
    "version": "1",
    "directory": "model_repository/adder/1",
    "backend": "deterministic_cpu"
  },
  "requests": []
}
```

Successful response:

```json
{
  "protocol": "infermesh-engine/1",
  "request_id": "17",
  "status": "ok",
  "responses": [],
  "execution_started": true
}
```

Errors include `error` and `execution_started`. Model import/session creation
errors report false; execution errors report true. Rust validates protocol
version, request ID, response count, and a per-call timeout. A timeout
terminates the worker and is conservatively classified post-execution.

Stdout is protocol-only. Framework diagnostics inherit stderr so they cannot
corrupt framing.

## Compatibility boundary

Protocol version 1 has no negotiation or binary tensor transport. Unknown
versions fail before execution. A future remote transport can preserve the
same envelopes while adding authentication, length-prefix framing, streaming,
and cancellation.
