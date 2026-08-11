import subprocess
import unittest
from pathlib import Path

from sgemm_tools.results import BEGIN_MARKER, END_MARKER


ROOT = Path(__file__).resolve().parents[1]


class DocumentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.results = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
        cls.gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    def test_readme_covers_every_phase_a_method(self):
        for method in (
            "naive",
            "coalesced",
            "shared",
            "blocktiling-1d",
            "blocktiling-2d",
            "vectorized",
            "autotuned",
            "warptiling",
            "cublas",
        ):
            self.assertIn(method, self.readme)

    def test_documented_python_options_exist(self):
        help_result = subprocess.run(
            ["python3", "benchmark.py", "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for option in (
            "--kernel",
            "--all",
            "--list",
            "--check-only",
            "--update-results",
            "--results-file",
            "--tune",
        ):
            self.assertIn(option, help_result.stdout)
            self.assertIn(option, self.readme)

    def test_results_has_one_managed_marker_pair(self):
        self.assertEqual(self.results.count(BEGIN_MARKER), 1)
        self.assertEqual(self.results.count(END_MARKER), 1)
        self.assertLess(self.results.index(BEGIN_MARKER), self.results.index(END_MARKER))

    def test_gitignore_covers_generated_artifacts_but_not_results(self):
        for pattern in (
            "/build/",
            "*.o",
            "*.exe",
            "*.exp",
            "*.lib",
            "__pycache__/",
            ".pytest_cache/",
            "*.tmp",
        ):
            self.assertIn(pattern, self.gitignore)
        self.assertNotIn("RESULTS.md", self.gitignore)

    def test_readme_explains_the_measurement_contract(self):
        for phrase in ("C = alpha * A * B + beta * C", "2*M*N*K + 2*M*N", "CUDA event"):
            self.assertIn(phrase, self.readme)


if __name__ == "__main__":
    unittest.main()
