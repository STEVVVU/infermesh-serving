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


class TorchAdapter:
    """Optional CPU TorchScript adapter; CUDA placement is intentionally absent."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def _model(self, descriptor: dict[str, Any]) -> tuple[Any, Any]:
        path = str(Path(descriptor["directory"]) / "model.pt")
        if path in self._models:
            return self._models[path]
        try:
            import torch
        except ImportError as error:
            raise AdapterError(
                "torch is not installed; install infermesh-worker[torch]",
                execution_started=False,
            ) from error
        if not Path(path).is_file():
            raise AdapterError(
                f"TorchScript model not found: {path}", execution_started=False
            )
        try:
            model = torch.jit.load(path, map_location="cpu").eval()
        except Exception as error:
            raise AdapterError(
                f"failed to load TorchScript model: {error}",
                execution_started=False,
            ) from error
        self._models[path] = (torch, model)
        return torch, model

    def infer(
        self, descriptor: dict[str, Any], requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        torch, model = self._model(descriptor)
        responses = []
        for request in requests:
            try:
                tensors = [
                    torch.tensor(
                        item["data"],
                        dtype=getattr(torch, _DTYPES[item["datatype"]]),
                    ).reshape(item["shape"])
                    for item in request["inputs"]
                ]
                with torch.inference_mode():
                    values = model(*tensors)
                if not isinstance(values, (tuple, list)):
                    values = [values]
                outputs = [
                    {
                        "name": f"OUTPUT{index}",
                        "shape": list(value.shape),
                        "datatype": _triton_dtype(str(value.dtype)),
                        "data": value.reshape(-1).tolist(),
                    }
                    for index, value in enumerate(values)
                ]
            except AdapterError:
                raise
            except Exception as error:
                raise AdapterError(
                    f"TorchScript execution failed: {error}",
                    execution_started=True,
                ) from error
            responses.append(
                {
                    "model_name": descriptor["name"],
                    "model_version": descriptor["version"],
                    "id": request.get("id", ""),
                    "outputs": outputs,
                    "parameters": {"backend": "torchscript_cpu"},
                }
            )
        return responses


def _triton_dtype(torch_dtype: str) -> str:
    normalized = torch_dtype.removeprefix("torch.")
    reverse = {value: key for key, value in _DTYPES.items()}
    if normalized not in reverse:
        raise AdapterError(
            f"unsupported Torch output dtype {torch_dtype}",
            execution_started=True,
        )
    return reverse[normalized]
