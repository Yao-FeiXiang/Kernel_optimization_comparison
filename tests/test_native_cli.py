import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCES = [
    ROOT / "src/benchmark.cu",
    ROOT / "src/reference.cpp",
    ROOT / "src/kernels/registry.cu",
    ROOT / "src/kernels/kernel_01_naive.cu",
    ROOT / "src/kernels/kernel_02_coalesced.cu",
    ROOT / "src/kernels/kernel_03_shared.cu",
    ROOT / "src/kernels/kernel_04_blocktiling_1d.cu",
    ROOT / "src/kernels/kernel_05_blocktiling_2d.cu",
    ROOT / "src/kernels/kernel_06_vectorized.cu",
    ROOT / "src/kernels/kernel_09_autotuned.cu",
    ROOT / "src/kernels/kernel_10_warptiling.cu",
    ROOT / "src/kernels/cublas_baseline.cu",
]


class NativeCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.executable = Path(cls.temp_directory.name) / "sgemm_bench"
        command = [
            "nvcc",
            "-std=c++17",
            f"-I{ROOT / 'include'}",
            *(str(path) for path in CUDA_SOURCES),
            "-o",
            str(cls.executable),
        ]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp_directory.cleanup()

    def run_cli(self, *arguments):
        return subprocess.run(
            [str(self.executable), *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_help_documents_supported_options(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--kernel",
            "--all",
            "--list",
            "--m",
            "--n",
            "--k",
            "--alpha",
            "--beta",
            "--warmup",
            "--repeat",
            "--seed",
            "--atol",
            "--rtol",
            "--check",
            "--check-only",
            "--json",
            "--tune",
        ):
            self.assertIn(option, result.stdout)

    def test_list_does_not_require_a_cuda_device(self):
        result = self.run_cli("--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [line.split()[0] for line in result.stdout.splitlines() if line],
            ["1", "2", "3", "4", "5", "6", "9", "10", "0"],
        )

    def test_unknown_kernel_is_a_usage_error(self):
        result = self.run_cli("--kernel", "missing")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown kernel", result.stderr.lower())

    def test_invalid_dimension_is_a_usage_error(self):
        result = self.run_cli("--kernel", "1", "--m", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("dimensions must be positive", result.stderr.lower())

    def test_kernel_and_all_are_mutually_exclusive(self):
        result = self.run_cli("--kernel", "1", "--all")
        self.assertEqual(result.returncode, 2)
        self.assertIn("mutually exclusive", result.stderr.lower())

    def test_tune_and_kernel_are_mutually_exclusive(self):
        result = self.run_cli("--tune", "--kernel", "1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("mutually exclusive", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
