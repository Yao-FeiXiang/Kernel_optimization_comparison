import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from benchmark import create_parser, main


class BenchmarkDefaultsTest(unittest.TestCase):
    def test_all_uses_tutorial_sized_defaults(self):
        arguments = create_parser().parse_args(["--all"])

        self.assertEqual((arguments.m, arguments.n, arguments.k), (4096, 4096, 4096))
        self.assertEqual(arguments.warmup, 5)
        self.assertEqual(arguments.repeat, 50)

    def test_explicit_values_override_defaults(self):
        arguments = create_parser().parse_args(
            [
                "--kernel",
                "10",
                "--m",
                "512",
                "--n",
                "768",
                "--k",
                "256",
                "--warmup",
                "2",
                "--repeat",
                "7",
            ]
        )

        self.assertEqual((arguments.m, arguments.n, arguments.k), (512, 768, 256))
        self.assertEqual(arguments.warmup, 2)
        self.assertEqual(arguments.repeat, 7)


class NoBuildTest(unittest.TestCase):
    def test_no_build_rejects_a_stale_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            build_dir = Path(directory)
            executable = build_dir / "sgemm_bench"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            stderr = io.StringIO()

            with (
                mock.patch("benchmark.executable_is_stale", create=True, return_value=True),
                redirect_stderr(stderr),
            ):
                return_code = main(["--all", "--no-build", "--build-dir", str(build_dir)])

        self.assertEqual(return_code, 1)
        self.assertIn("stale", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
