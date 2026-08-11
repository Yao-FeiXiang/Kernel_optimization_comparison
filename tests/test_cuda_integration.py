import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sgemm_tools.runner import NativeRunOptions, build_executable, native_arguments


ROOT = Path(__file__).resolve().parents[1]


def gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    probe = subprocess.run(
        ["nvidia-smi", "-L"], check=False, capture_output=True, text=True
    )
    return probe.returncode == 0


class CudaIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("nvcc") is None:
            raise unittest.SkipTest("nvcc is unavailable")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.executable = build_executable(
            ROOT, Path(cls.temporary.name), force=True, prefer_cmake=False
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def test_list_contains_all_phase_a_methods_without_a_driver(self):
        result = subprocess.run(
            [str(self.executable), "--list"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.strip().splitlines()), 7)

    @unittest.skipUnless(gpu_available(), "CUDA driver/device unavailable")
    def test_all_kernels_match_cpu_reference_on_tail_shape(self):
        arguments = native_arguments(
            NativeRunOptions(all=True, m=37, n=41, k=29, check_only=True, json=True)
        )
        result = subprocess.run(
            [str(self.executable), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(result.stdout.count('"status":"pass"'), 7)

    @unittest.skipUnless(gpu_available(), "CUDA driver/device unavailable")
    def test_vectorized_method_reports_float4_and_scalar_fallback(self):
        variants = []
        for m, n, k in ((128, 128, 64), (131, 127, 35)):
            arguments = native_arguments(
                NativeRunOptions(
                    kernel="vectorized", m=m, n=n, k=k, check_only=True, json=True
                )
            )
            result = subprocess.run(
                [str(self.executable), *arguments],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            variants.append(result.stdout)
        self.assertIn('"implementation_variant":"float4"', variants[0])
        self.assertIn('"implementation_variant":"scalar-fallback"', variants[1])


if __name__ == "__main__":
    unittest.main()
