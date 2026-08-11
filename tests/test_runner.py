import json
import tempfile
import unittest
from pathlib import Path

from sgemm_tools.runner import (
    NativeRunOptions,
    ResultParseError,
    native_arguments,
    nvcc_command,
    parse_json_lines,
)


ROOT = Path(__file__).resolve().parents[1]


class NativeArgumentsTest(unittest.TestCase):
    def test_single_kernel_arguments_forward_all_values(self):
        options = NativeRunOptions(
            kernel="blocktiling-2d",
            m=37,
            n=41,
            k=29,
            alpha=0.75,
            beta=0.25,
            warmup=3,
            repeat=7,
            seed=9,
            atol=0.01,
            rtol=0.02,
            check=True,
            check_only=False,
            json=True,
        )
        arguments = native_arguments(options)
        self.assertEqual(arguments[:2], ["--kernel", "blocktiling-2d"])
        for expected in (
            ["--m", "37"],
            ["--n", "41"],
            ["--k", "29"],
            ["--alpha", "0.75"],
            ["--beta", "0.25"],
            ["--warmup", "3"],
            ["--repeat", "7"],
            ["--seed", "9"],
            ["--atol", "0.01"],
            ["--rtol", "0.02"],
        ):
            position = arguments.index(expected[0])
            self.assertEqual(arguments[position : position + 2], expected)
        self.assertIn("--check", arguments)
        self.assertIn("--json", arguments)

    def test_all_check_only_arguments(self):
        arguments = native_arguments(NativeRunOptions(all=True, check_only=True))
        self.assertEqual(arguments[0], "--all")
        self.assertIn("--check-only", arguments)


class JsonLineTest(unittest.TestCase):
    def test_human_lines_are_ignored_and_json_is_parsed(self):
        payload = {
            "schema_version": 1,
            "method_id": 1,
            "method": "naive",
            "status": "pass",
            "m": 1,
            "n": 1,
            "k": 1,
            "alpha": 0.8,
            "beta": 0.2,
            "warmup": 1,
            "repeat": 1,
            "latency_ms": 0.1,
            "min_latency_ms": 0.1,
            "gflops": 0.00004,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "gpu": "Test",
            "compute_capability": "8.6",
            "cuda_runtime": "13.3",
            "timestamp_utc": "2026-08-11T00:00:00Z",
            "message": "ignored forward-compatible field",
        }
        output = "[1] naive: pass\n" + json.dumps(payload) + "\n"
        records = parse_json_lines(output)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].method, "naive")

    def test_malformed_json_line_is_rejected(self):
        with self.assertRaises(ResultParseError):
            parse_json_lines("human output\n{not-json}\n")


class BuildCommandTest(unittest.TestCase):
    def test_nvcc_command_contains_every_phase_a_source(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sgemm_bench"
            command = nvcc_command(ROOT, output)
        self.assertEqual(command[0], "nvcc")
        self.assertIn("-std=c++17", command)
        self.assertEqual(command[-2:], ["-o", str(output)])
        for source in (
            "src/benchmark.cu",
            "src/reference.cpp",
            "src/kernels/registry.cu",
            "src/kernels/kernel_01_naive.cu",
            "src/kernels/kernel_02_coalesced.cu",
            "src/kernels/kernel_03_shared.cu",
            "src/kernels/kernel_04_blocktiling_1d.cu",
            "src/kernels/kernel_05_blocktiling_2d.cu",
            "src/kernels/kernel_06_vectorized.cu",
            "src/kernels/kernel_09_autotuned.cu",
        ):
            self.assertIn(str(ROOT / source), command)

    def test_tune_selects_the_autotuned_native_method(self):
        arguments = native_arguments(NativeRunOptions(tune=True))
        self.assertEqual(arguments[0], "--tune")


if __name__ == "__main__":
    unittest.main()
