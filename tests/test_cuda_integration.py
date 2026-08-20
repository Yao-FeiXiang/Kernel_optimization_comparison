import json
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
        self.assertEqual(len(result.stdout.strip().splitlines()), 9)

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
        self.assertGreaterEqual(result.stdout.count('"status":"pass"'), 8)
        self.assertEqual(
            result.stdout.count('"status":"pass"')
            + result.stdout.count('"status":"unavailable"'),
            9,
        )

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

    @unittest.skipUnless(gpu_available(), "CUDA driver/device unavailable")
    def test_tutorial_aligned_paths_report_fast_variants(self):
        expected_variants = {
            "blocktiling-2d": "register-tile-2d-fast",
            "autotuned": "autotuned-vectorized",
            "warptiling": "warp-tiled-a100",
        }
        for method, expected_variant in expected_variants.items():
            with self.subTest(method=method):
                arguments = native_arguments(
                    NativeRunOptions(
                        kernel=method,
                        m=128,
                        n=128,
                        k=64,
                        check_only=True,
                        json=True,
                    )
                )
                result = subprocess.run(
                    [str(self.executable), *arguments],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                record = next(
                    json.loads(line)
                    for line in result.stdout.splitlines()
                    if line.startswith("{")
                )
                self.assertEqual(record["status"], "pass")
                self.assertEqual(record["implementation_variant"], expected_variant)

    @unittest.skipUnless(gpu_available(), "CUDA driver/device unavailable")
    def test_coalescing_stage_improves_large_square(self):
        measurements = {}
        for method in ("naive", "coalesced"):
            arguments = native_arguments(
                NativeRunOptions(
                    kernel=method,
                    m=1024,
                    n=1024,
                    k=1024,
                    warmup=3,
                    repeat=20,
                    check=True,
                    json=True,
                )
            )
            result = subprocess.run(
                [str(self.executable), *arguments],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            record = next(
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith("{")
            )
            self.assertEqual(record["status"], "pass")
            measurements[method] = record["gflops"]

        self.assertGreater(
            measurements["coalesced"], measurements["naive"] * 1.5
        )


if __name__ == "__main__":
    unittest.main()
