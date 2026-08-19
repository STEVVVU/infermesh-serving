from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from infermesh_worker.adapters.cpu import CpuAdapter
from infermesh_worker.adapters.onnx_runtime import OnnxRuntimeAdapter
from infermesh_worker.adapters.torch_adapter import TorchAdapter
from infermesh_worker.errors import AdapterError


PROTOCOL = "infermesh-engine/1"
MAX_LINE_BYTES = 8 * 1024 * 1024


def adapter_for(name: str) -> Any:
    if name == "cpu":
        return CpuAdapter()
    if name == "onnxruntime":
        return OnnxRuntimeAdapter()
    if name == "torch":
        return TorchAdapter()
    raise ValueError(f"unknown backend {name}")


def handle(adapter: Any, envelope: dict[str, Any]) -> dict[str, Any]:
    request_id = str(envelope.get("request_id", ""))
    if envelope.get("protocol") != PROTOCOL:
        return failure(request_id, "protocol version mismatch", False)
    try:
        responses = adapter.infer(envelope["model"], envelope["requests"])
    except AdapterError as error:
        return failure(request_id, str(error), error.execution_started)
    except (KeyError, TypeError, ValueError) as error:
        return failure(request_id, f"invalid engine request: {error}", False)
    return {
        "protocol": PROTOCOL,
        "request_id": request_id,
        "status": "ok",
        "responses": responses,
        "execution_started": True,
    }


def failure(request_id: str, message: str, execution_started: bool) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "request_id": request_id,
        "status": "error",
        "error": message,
        "execution_started": execution_started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["cpu", "onnxruntime", "torch"], required=True
    )
    arguments = parser.parse_args()
    adapter = adapter_for(arguments.backend)
    for line in sys.stdin.buffer:
        if len(line) > MAX_LINE_BYTES:
            response = failure("", "engine request exceeds line limit", False)
        else:
            try:
                envelope = json.loads(line)
                response = handle(adapter, envelope)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                response = failure("", f"invalid JSON: {error}", False)
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
