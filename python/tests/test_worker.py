from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from infermesh_worker.adapters.cpu import CpuAdapter
from infermesh_worker.worker import PROTOCOL, handle


def envelope() -> dict:
    return {
        "protocol": PROTOCOL,
        "request_id": "test-1",
        "model": {
            "name": "adder",
            "version": "1",
            "directory": ".",
            "backend": "deterministic_cpu",
        },
        "requests": [
            {
                "id": "request",
                "parameters": {},
                "inputs": [
                    {
                        "name": "INPUT0",
                        "shape": [2, 1],
                        "datatype": "FP64",
                        "data": [1.0, 2.0],
                    },
                    {
                        "name": "INPUT1",
                        "shape": [2, 1],
                        "datatype": "FP64",
                        "data": [3.0, 4.0],
                    },
                ],
                "outputs": [],
            }
        ],
    }


class WorkerTest(unittest.TestCase):
    def test_cpu_adapter_protocol_response(self) -> None:
        response = handle(CpuAdapter(), envelope())
        self.assertEqual("ok", response["status"])
        self.assertEqual(
            [4.0, 6.0], response["responses"][0]["outputs"][0]["data"]
        )
        self.assertTrue(response["execution_started"])

    def test_subprocess_line_protocol(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.path.dirname(os.path.dirname(__file__))
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "infermesh_worker.worker",
                "--backend",
                "cpu",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=environment,
        )
        output, _ = process.communicate(json.dumps(envelope()) + "\n", timeout=10)
        self.assertEqual(0, process.returncode)
        response = json.loads(output)
        self.assertEqual("test-1", response["request_id"])
        self.assertEqual([4.0, 6.0], response["responses"][0]["outputs"][0]["data"])


if __name__ == "__main__":
    unittest.main()
