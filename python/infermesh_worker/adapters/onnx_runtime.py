from __future__ import annotations

from pathlib import Path
from typing import Any

from infermesh_worker.errors import AdapterError


_DTYPES = {
    "FP32": "float32",
    "FP64": "float64",
    "INT32": "int32",
    "INT64": "int64",
}


class OnnxRuntimeAdapter:
    """Real ONNX Runtime adapter, loaded only when this backend is selected."""

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}

    def _session(self, model: dict[str, Any]) -> Any:
        model_path = str(Path(model["directory"]) / "model.onnx")
        if model_path in self._sessions:
            return self._sessions[model_path]
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise AdapterError(
                "onnxruntime is not installed; install infermesh-worker[onnx]",
                execution_started=False,
            ) from error
        if not Path(model_path).is_file():
            raise AdapterError(
                f"ONNX model not found: {model_path}", execution_started=False
            )
        try:
            session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise AdapterError(
                f"failed to load ONNX model: {error}", execution_started=False
            ) from error
        self._sessions[model_path] = session
        return session

    def infer(
        self, model: dict[str, Any], requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        session = self._session(model)
        try:
            import numpy as np
        except ImportError as error:
            raise AdapterError(
                "numpy is not installed; install infermesh-worker[onnx]",
                execution_started=False,
            ) from error

        responses = []
        for request in requests:
            try:
                feeds = {
                    tensor["name"]: np.asarray(
                        tensor["data"], dtype=_DTYPES[tensor["datatype"]]
                    ).reshape(tensor["shape"])
                    for tensor in request["inputs"]
                }
                output_names = (
                    [item["name"] for item in request.get("outputs", [])]
                    or [item.name for item in session.get_outputs()]
                )
                values = session.run(output_names, feeds)
                outputs = [
                    {
                        "name": name,
                        "shape": list(value.shape),
                        "datatype": _triton_dtype(str(value.dtype)),
                        "data": value.reshape(-1).tolist(),
                    }
                    for name, value in zip(output_names, values)
                ]
            except AdapterError:
                raise
            except Exception as error:
                raise AdapterError(
                    f"ONNX execution failed: {error}", execution_started=True
                ) from error
            responses.append(
                {
                    "model_name": model["name"],
                    "model_version": model["version"],
                    "id": request.get("id", ""),
                    "outputs": outputs,
                    "parameters": {
                        "backend": "onnxruntime",
                        "provider": "CPUExecutionProvider",
                    },
                }
            )
        return responses


def _triton_dtype(numpy_dtype: str) -> str:
    reverse = {value: key for key, value in _DTYPES.items()}
    if numpy_dtype not in reverse:
        raise AdapterError(
            f"unsupported ONNX output dtype {numpy_dtype}",
            execution_started=True,
        )
    return reverse[numpy_dtype]
