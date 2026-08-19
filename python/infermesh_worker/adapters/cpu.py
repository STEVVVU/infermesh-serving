from __future__ import annotations

from typing import Any

from infermesh_worker.errors import AdapterError


class CpuAdapter:
    """Deterministic dependency-free backend used for tests and local runs."""

    def infer(
        self, model: dict[str, Any], requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        responses = []
        for request in requests:
            inputs = request.get("inputs", [])
            if not inputs:
                raise AdapterError(
                    "CPU adapter requires an input", execution_started=True
                )
            if model["name"] == "adder":
                if len(inputs) != 2:
                    raise AdapterError(
                        "adder requires two inputs", execution_started=True
                    )
                left = inputs[0]["data"]
                right = inputs[1]["data"]
                if len(left) != len(right):
                    raise AdapterError(
                        "adder input lengths differ", execution_started=True
                    )
                data = [float(a) + float(b) for a, b in zip(left, right)]
                output_name = "OUTPUT0"
                datatype = "FP64"
            else:
                data = list(inputs[0]["data"])
                output_name = inputs[0]["name"]
                datatype = inputs[0]["datatype"]
            responses.append(
                {
                    "model_name": model["name"],
                    "model_version": model["version"],
                    "id": request.get("id", ""),
                    "outputs": [
                        {
                            "name": output_name,
                            "shape": inputs[0]["shape"],
                            "datatype": datatype,
                            "data": data,
                        }
                    ],
                    "parameters": {"backend": "python_cpu"},
                }
            )
        return responses
