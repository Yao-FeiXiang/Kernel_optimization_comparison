# Simple Default Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `python3 benchmark.py --all --update-results` run the recommended A100/4096 experiment by default, automatically rebuild stale native code, and document the workflow with a short README.

**Architecture:** Keep the Python wrapper as the user-facing policy layer: shared Python constants define the recommended matrix and timing defaults, while the native CLI retains its lightweight defaults. Add one filesystem-only freshness function in `sgemm_tools.runner` and use it from both normal build reuse and explicit `--no-build` validation. Preserve the existing benchmark/result pipeline and only simplify its documentation.

**Tech Stack:** Python 3 standard library (`argparse`, `pathlib`, `unittest`), CUDA/NVCC native executable, Markdown results and README.

---

### Task 1: Set the Python entry point's tutorial defaults

**Files:**

- Create: `tests/test_benchmark.py`
- Modify: `sgemm_tools/runner.py`
- Modify: `benchmark.py`

- [ ] **Step 1: Write failing default and override tests**

Create `tests/test_benchmark.py`:

```python
import unittest

from benchmark import create_parser


class BenchmarkDefaultsTest(unittest.TestCase):
    def test_all_uses_tutorial_comparable_defaults(self):
        arguments = create_parser().parse_args(["--all"])
        self.assertEqual((arguments.m, arguments.n, arguments.k), (4096, 4096, 4096))
        self.assertEqual(arguments.warmup, 5)
        self.assertEqual(arguments.repeat, 50)

    def test_explicit_problem_and_timing_override_defaults(self):
        arguments = create_parser().parse_args(
            [
                "--kernel", "10", "--m", "512", "--n", "768", "--k", "256",
                "--warmup", "2", "--repeat", "7",
            ]
        )
        self.assertEqual((arguments.m, arguments.n, arguments.k), (512, 768, 256))
        self.assertEqual(arguments.warmup, 2)
        self.assertEqual(arguments.repeat, 7)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_benchmark -v
```

Expected: `test_all_uses_tutorial_comparable_defaults` fails with the current `(1024, 1024, 1024)` and repeat `20`; the explicit override test passes.

- [ ] **Step 3: Define shared Python defaults and use them in both layers**

Add near the top of `sgemm_tools/runner.py`:

```python
DEFAULT_MATRIX_SIZE = 4096
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 50
```

Use them in `NativeRunOptions`:

```python
m: int = DEFAULT_MATRIX_SIZE
n: int = DEFAULT_MATRIX_SIZE
k: int = DEFAULT_MATRIX_SIZE
warmup: int = DEFAULT_WARMUP
repeat: int = DEFAULT_REPEAT
```

Import the constants in `benchmark.py` and use them as the `argparse` defaults for `--m`, `--n`, `--k`, `--warmup`, and `--repeat`. Do not change the native C++ defaults because the wrapper always forwards explicit values.

- [ ] **Step 4: Verify GREEN and existing argument forwarding**

Run:

```bash
python3 -m unittest tests.test_benchmark tests.test_runner -v
```

Expected: all tests pass and explicit argument forwarding remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add benchmark.py sgemm_tools/runner.py tests/test_benchmark.py
git commit -m "feat: use tutorial benchmark defaults"
```

### Task 2: Rebuild stale native executables automatically

**Files:**

- Modify: `sgemm_tools/runner.py`
- Modify: `benchmark.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing freshness tests**

Add imports for `os` and the runner module (`from sgemm_tools import runner`) as required. Add to `tests/test_runner.py`:

```python
class BuildFreshnessTest(unittest.TestCase):
    def test_runner_exposes_build_freshness_check(self):
        self.assertTrue(hasattr(runner, "executable_is_stale"))

    def test_executable_is_stale_when_a_header_is_newer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / "include" / "sgemm" / "example.hpp"
            executable = root / "build" / "sgemm_bench"
            header.parent.mkdir(parents=True)
            executable.parent.mkdir(parents=True)
            header.write_text("#pragma once\n", encoding="utf-8")
            executable.touch()
            os.utime(executable, ns=(1_000_000_000, 1_000_000_000))
            os.utime(header, ns=(2_000_000_000, 2_000_000_000))
            self.assertTrue(runner.executable_is_stale(root, executable))

    def test_executable_is_fresh_when_it_is_newer_than_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "benchmark.cu"
            executable = root / "build" / "sgemm_bench"
            source.parent.mkdir(parents=True)
            executable.parent.mkdir(parents=True)
            source.write_text("// source\n", encoding="utf-8")
            executable.touch()
            os.utime(source, ns=(1_000_000_000, 1_000_000_000))
            os.utime(executable, ns=(2_000_000_000, 2_000_000_000))
            self.assertFalse(runner.executable_is_stale(root, executable))
```

First add only `test_runner_exposes_build_freshness_check`, run it, and observe a normal assertion failure. After the function exists, add the two mtime behavior tests above plus this `tests/test_benchmark.py` case:

```python
def test_no_build_rejects_stale_executable(self):
    with tempfile.TemporaryDirectory() as directory:
        build_dir = Path(directory)
        executable = build_dir / "sgemm_bench"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        with mock.patch(
            "benchmark.executable_is_stale", create=True, return_value=True
        ):
            stderr = StringIO()
            with redirect_stderr(stderr):
                status = main(["--list", "--no-build", "--build-dir", str(build_dir)])
    self.assertEqual(status, 1)
    self.assertIn("stale", stderr.getvalue())
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_runner.BuildFreshnessTest tests.test_benchmark.BenchmarkDefaultsTest.test_no_build_rejects_stale_executable -v
```

Expected: `test_runner_exposes_build_freshness_check` fails because the runner has no freshness API. After adding the API but before updating `benchmark.py`, the `--no-build` test fails with status `0` instead of `1`.

- [ ] **Step 3: Implement filesystem-only build freshness**

Add to `sgemm_tools/runner.py`:

```python
def _build_inputs(root: Path) -> list[Path]:
    inputs = [root / source for source in PHASE_A_SOURCES]
    inputs.append(root / "CMakeLists.txt")
    include = root / "include"
    if include.exists():
        inputs.extend(path for path in include.rglob("*") if path.is_file())
    return [path for path in inputs if path.exists()]


def executable_is_stale(root: Path, executable: Path) -> bool:
    if not executable.exists():
        return True
    executable_mtime = executable.stat().st_mtime_ns
    return any(path.stat().st_mtime_ns > executable_mtime for path in _build_inputs(root))
```

Change the early return in `build_executable` to:

```python
if executable.exists() and not force and not executable_is_stale(root, executable):
    return executable
```

- [ ] **Step 4: Reject stale `--no-build` programs**

Import `executable_is_stale` in `benchmark.py` and change the `--no-build` branch to:

```python
if arguments.no_build:
    if not executable.exists():
        raise BuildError(f"native executable does not exist: {executable}")
    if executable_is_stale(ROOT, executable):
        raise BuildError(
            f"native executable is stale: {executable}; "
            "remove --no-build or use --build"
        )
```

- [ ] **Step 5: Verify GREEN and normal build command construction**

Run:

```bash
python3 -m unittest tests.test_runner tests.test_benchmark -v
```

Expected: all freshness, default, override, and existing runner tests pass.

- [ ] **Step 6: Commit**

```bash
git add benchmark.py sgemm_tools/runner.py tests/test_benchmark.py tests/test_runner.py
git commit -m "fix: rebuild stale benchmark executables"
```

### Task 3: Replace the README command catalog with a short workflow

**Files:**

- Modify: `README.md`
- Modify: `tests/test_docs.py`

- [ ] **Step 1: Write failing documentation simplicity tests**

Add to `tests/test_docs.py`:

```python
def test_readme_leads_with_simple_default_command(self):
    command = "python3 benchmark.py --all --update-results"
    self.assertIn(command, self.readme)
    self.assertLess(self.readme.index(command), self.readme.index("手动覆盖"))

def test_readme_keeps_shell_examples_compact(self):
    self.assertLessEqual(self.readme.count("```bash"), 6)
```

Update `test_readme_documents_tutorial_comparable_a100_run` so it checks the documented default values in prose (`4096×4096×4096`, `warmup=5`, `repeat=50`) rather than requiring the long full command.

- [ ] **Step 2: Run documentation tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_docs -v
```

Expected: the simple default command is absent and the current README has more than six shell code blocks.

- [ ] **Step 3: Rewrite README with four command workflows**

Keep the title, one-paragraph purpose, SGEMM formula, and method table. Immediately after the table document:

```markdown
## 直接运行

默认实验是 A100 教程对比口径：4096×4096×4096、warmup=5、repeat=50。程序默认编译 `sm_80`，源码或头文件更新后会自动重新构建。

```bash
python3 benchmark.py --all --update-results
```
```

Then include only these additional workflows:

```bash
# 单方法或自动调优
python3 benchmark.py --kernel 10 --update-results
python3 benchmark.py --tune --update-results
```

```bash
# 手动覆盖默认尺寸和轮数
python3 benchmark.py --all --m 2048 --n 2048 --k 2048 --repeat 20 --update-results
```

```bash
# 小尺寸正确性检查
python3 benchmark.py --all --m 37 --n 41 --k 29 --check-only
```

Explain remaining options (`--list`, `--results-file`, `--build`, `--no-build`, `SGEMM_CUDA_ARCH`) as one compact bullet list without more shell blocks. Retain the timing contract phrases `C = alpha * A * B + beta * C`, `2*M*N*K + 2*M*N`, `CUDA event`, `Performance vs cuBLAS`, `GFLOP/s`, `register-tile-2d-fast`, `tail-safe`, and `GPU0` to preserve existing documentation contracts. Remove the separate CMake walkthrough, add-kernel guide, repeated 1024 commands, and long troubleshooting command catalog.

- [ ] **Step 4: Verify README and result documentation tests**

Run:

```bash
python3 -m unittest tests.test_docs tests.test_results -v
```

Expected: all tests pass and README contains no more than six shell code blocks.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_docs.py
git commit -m "docs: simplify benchmark workflow"
```

### Task 4: Correct stored results with the fresh build and complete verification

**Files:**

- Modify: `RESULTS.md` (generated by `benchmark.py`)

- [ ] **Step 1: Check GPU0 and run the new default command end-to-end**

Run:

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free --format=csv,noheader
CUDA_VISIBLE_DEVICES=0 python3 benchmark.py --all --update-results
```

Expected: GPU0 is idle before launch; the command uses 4096/5/50, automatically builds the current sources when necessary, reports all nine methods, and updates the existing 4096 group.

- [ ] **Step 2: Correct the user's latest 1024 group with the fresh executable**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 benchmark.py --all --m 1024 --n 1024 --k 1024 --warmup 5 --repeat 20 --no-build --update-results
```

Expected: the 1024 group now reports `register-tile-2d-fast`, `autotuned-vectorized`, and `warp-tiled-a100` rather than the stale variants, while the 4096 group remains present.

- [ ] **Step 3: Run the complete verification suite**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m unittest discover -s tests -v
git diff --check
git status --short
```

Expected: every Python/CUDA test passes, the diff has no whitespace errors, and only the generated `RESULTS.md` remains uncommitted.

- [ ] **Step 4: Inspect the generated result contracts**

Run:

```bash
rg -n "M=1024|M=4096|register-tile-2d-fast|autotuned-vectorized|warp-tiled-a100" RESULTS.md
```

Expected: both matrix-size groups and all current fast-path variants are present.

- [ ] **Step 5: Commit generated results**

```bash
git add RESULTS.md
git commit -m "results: refresh benchmark defaults"
```

### Task 5: Review and integrate

- [ ] **Step 1: Review the full feature diff**

Run:

```bash
git diff --check main...HEAD
git diff --stat main...HEAD
git status --short
```

Expected: only the approved defaults, freshness protection, concise documentation, tests, and generated results changed; worktree is clean.

- [ ] **Step 2: Re-run the completion gate**

Run the full Python/CUDA test command again with GPU0 and record the exact passing count.

- [ ] **Step 3: Merge and clean up automatically**

Fast-forward the verified feature branch into `main`, run the focused defaults/freshness/docs tests on merged `main`, then remove the temporary worktree and delete the merged branch. Do not push a remote branch.
