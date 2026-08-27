from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper

from infermesh_worker.adapters.onnx_runtime import OnnxRuntimeAdapter


class OnnxRuntimeIntegrationTest(unittest.TestCase):
    def test_generated_add_model_runs_named_inputs_and_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="infermesh-onnx-") as directory:
            model_path = Path(directory) / "model.onnx"
            graph = helper.make_graph(
                [helper.make_node("Add", ["INPUT0", "INPUT1"], ["SUM"])],
                "infermesh_add",
                [
                    helper.make_tensor_value_info(
                        "INPUT0", TensorProto.FLOAT, [2, 2]
                    ),
                    helper.make_tensor_value_info(
                        "INPUT1", TensorProto.FLOAT, [2, 2]
                    ),
                ],
                [
                    helper.make_tensor_value_info(
                        "SUM", TensorProto.FLOAT, [2, 2]
                    )
                ],
            )
            model = helper.make_model(
                graph,
                producer_name="infermesh-test",
                opset_imports=[helper.make_opsetid("", 13)],
            )
            # ONNX Runtime supports this stable IR version across the tested range.
            model.ir_version = 8
            checker.check_model(model)
            onnx.save(model, model_path)

            response = OnnxRuntimeAdapter().infer(
                {
                    "name": "generated-add",
                    "version": "1",
                    "directory": directory,
                    "backend": "onnxruntime",
                },
                [
                    {
                        "id": "onnx-integration",
                        "inputs": [
                            {
                                "name": "INPUT0",
                                "shape": [2, 2],
                                "datatype": "FP32",
                                "data": [1.0, 2.0, 3.0, 4.0],
                            },
                            {
                                "name": "INPUT1",
                                "shape": [2, 2],
                                "datatype": "FP32",
                                "data": [10.0, 20.0, 30.0, 40.0],
                            },
                        ],
                        "outputs": [{"name": "SUM"}],
                    }
                ],
            )[0]

            output = response["outputs"][0]
            self.assertEqual("SUM", output["name"])
            self.assertEqual([2, 2], output["shape"])
            self.assertEqual("FP32", output["datatype"])
            self.assertEqual([11.0, 22.0, 33.0, 44.0], output["data"])
            self.assertEqual(
                "CPUExecutionProvider", response["parameters"]["provider"]
            )


if __name__ == "__main__":
    unittest.main()
