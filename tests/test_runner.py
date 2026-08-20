import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sgemm_tools import runner
from sgemm_tools.runner import (
    NativeRunOptions,
    ResultParseError,
    native_arguments,
    nvcc_command,
    parse_json_lines,
)


ROOT = Path(__file__).resolve().parents[1]


class BuildFreshnessTest(unittest.TestCase):
    def test_runner_exposes_build_freshness_check(self):
        self.assertTrue(hasattr(runner, "executable_is_stale"))

    def test_newer_header_marks_executable_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / "include" / "sgemm" / "kernel.cuh"
            executable = root / "build" / "sgemm_bench"
            header.parent.mkdir(parents=True)
            executable.parent.mkdir(parents=True)
            header.touch()
            executable.touch()
            os.utime(executable, ns=(1_000_000_000, 1_000_000_000))
            os.utime(header, ns=(2_000_000_000, 2_000_000_000))

            self.assertTrue(runner.executable_is_stale(root, executable))

    def test_older_source_keeps_executable_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "benchmark.cu"
            executable = root / "build" / "sgemm_bench"
            source.parent.mkdir(parents=True)
            executable.parent.mkdir(parents=True)
            source.touch()
            executable.touch()
            os.utime(source, ns=(1_000_000_000, 1_000_000_000))
            os.utime(executable, ns=(2_000_000_000, 2_000_000_000))

            self.assertFalse(runner.executable_is_stale(root, executable))

    def test_stale_executable_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dir = root / "build"
            executable = build_dir / "sgemm_bench"
            build_dir.mkdir()
            executable.touch()

            with (
                mock.patch.object(runner, "executable_is_stale", return_value=True),
                mock.patch.object(runner.shutil, "which", return_value="/usr/bin/nvcc"),
                mock.patch.object(runner, "_run_build") as run_build,
            ):
                result = runner.build_executable(root, build_dir, prefer_cmake=False)

            self.assertEqual(result, executable)
            run_build.assert_called_once()


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
            "src/kernels/kernel_10_warptiling.cu",
            "src/kernels/cublas_baseline.cu",
        ):
            self.assertIn(str(ROOT / source), command)

    def test_tune_selects_the_autotuned_native_method(self):
        arguments = native_arguments(NativeRunOptions(tune=True))
        self.assertEqual(arguments[0], "--tune")

    def test_cublas_library_is_linked_after_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda_root = Path(directory) / "cuda"
            library = cuda_root / "lib64" / "libcublas.so"
            library.parent.mkdir(parents=True)
            library.touch()
            with mock.patch.dict(os.environ, {"CUDA_HOME": str(cuda_root)}):
                command = nvcc_command(ROOT, Path(directory) / "sgemm_bench")
        self.assertGreater(
            command.index("-lcublas"),
            command.index(str(ROOT / "src/kernels/cublas_baseline.cu")),
        )

    def test_nvcc_command_targets_a100_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {}, clear=True):
                command = nvcc_command(ROOT, Path(directory) / "sgemm_bench")
        self.assertIn("-arch=sm_80", command)

    def test_nvcc_command_allows_cuda_arch_override(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"SGEMM_CUDA_ARCH": "sm_90"}):
                command = nvcc_command(ROOT, Path(directory) / "sgemm_bench")
        self.assertIn("-arch=sm_90", command)
        self.assertNotIn("-arch=sm_80", command)


if __name__ == "__main__":
    unittest.main()
