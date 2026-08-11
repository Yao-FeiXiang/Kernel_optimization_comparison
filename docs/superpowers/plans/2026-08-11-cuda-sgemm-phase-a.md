# CUDA SGEMM Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a modular, correctness-checked benchmark for tutorial kernels 1–5 with single/batch execution and idempotent Markdown result updates.

**Architecture:** Separate each CUDA kernel and its launch wrapper from a shared benchmark runner. The native executable owns CUDA allocation, correctness checks, CUDA-event timing, and JSON output; a dependency-free Python CLI owns building, orchestration, and `RESULTS.md` generation.

**Tech Stack:** CUDA C++17 / CUDA Runtime API, CMake 3.24+, Python 3.9+ standard library, `unittest`.

---

## File map

- Create `CMakeLists.txt`: formal CUDA build and test targets.
- Create `include/sgemm/benchmark.hpp`: option/result/value types and CPU-testable declarations.
- Create `include/sgemm/kernels.cuh`: stable kernel registry and launch-wrapper declarations.
- Create `src/reference.cpp`: deterministic inputs, CPU SGEMM, tolerance and FLOP helpers.
- Create `src/kernels/kernel_01_naive.cu`: tutorial Kernel 1.
- Create `src/kernels/kernel_02_coalesced.cu`: tutorial Kernel 2.
- Create `src/kernels/kernel_03_shared.cu`: tutorial Kernel 3.
- Create `src/kernels/kernel_04_blocktiling_1d.cu`: tutorial Kernel 4.
- Create `src/kernels/kernel_05_blocktiling_2d.cu`: tutorial Kernel 5.
- Create `src/kernels/registry.cu`: method metadata, aliases and wrapper lookup.
- Create `src/benchmark.cu`: native CLI, CUDA resource lifecycle, validation, timing and JSON Lines output.
- Create `tests/reference_test.cpp`: CPU-only native helper tests.
- Create `sgemm_tools/results.py`: result schema, derived metrics and Markdown rendering/upsert.
- Create `sgemm_tools/runner.py`: compiler detection, native command construction and JSON parsing.
- Create `sgemm_tools/__init__.py`: package marker and public exports.
- Create `benchmark.py`: user-facing Python CLI.
- Create `tests/test_results.py`: result parser and Markdown update tests.
- Create `tests/test_runner.py`: selection/build-command tests.
- Create `tests/test_cuda_integration.py`: conditional CUDA compile/run tests.
- Create `RESULTS.md`: tracked generated benchmark report.
- Create `.gitignore`: build and local artifact policy.
- Create `README.md`: Chinese-first usage and teaching documentation.
- Preserve `SGEMM.cu`, `script.py`, `SGEMM.exe`, `SGEMM.exp`, and `SGEMM.lib` as legacy inputs during Phase A.

### Task 1: Result record and Markdown update core

**Files:**
- Create: `sgemm_tools/__init__.py`
- Create: `sgemm_tools/results.py`
- Test: `tests/test_results.py`

- [ ] **Step 1: Write failing parser and idempotency tests**

Define fixtures with the exact native JSON schema and assert parsing, stable experiment keys, replacement of an existing method row, retention of another matrix-size section, and atomic file output:

```python
SAMPLE = {
    "schema_version": 1,
    "method_id": 1,
    "method": "naive",
    "status": "pass",
    "m": 64,
    "n": 65,
    "k": 33,
    "alpha": 0.8,
    "beta": 0.2,
    "warmup": 2,
    "repeat": 5,
    "latency_ms": 0.125,
    "min_latency_ms": 0.120,
    "gflops": 2.21,
    "max_abs_error": 0.0001,
    "max_rel_error": 0.00001,
    "gpu": "Test GPU",
    "compute_capability": "8.6",
    "cuda_runtime": "13.0",
    "timestamp_utc": "2026-08-11T00:00:00Z",
}

def test_upsert_replaces_same_configuration_and_method(self):
    first = BenchmarkRecord.from_mapping(SAMPLE)
    faster = BenchmarkRecord.from_mapping({**SAMPLE, "gflops": 3.0})
    markdown = render_results(upsert_records([], [first]))
    records = parse_managed_records(markdown)
    updated = render_results(upsert_records(records, [faster]))
    self.assertEqual(updated.count("| 1 | naive |"), 1)
    self.assertIn("| 3.000 |", updated)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python3 -m unittest tests.test_results -v`

Expected: import failure for missing `sgemm_tools.results`.

- [ ] **Step 3: Implement the result model and deterministic renderer**

Implement a frozen `BenchmarkRecord` dataclass with `from_mapping`, an experiment key `(gpu, compute_capability, cuda_runtime, m, n, k, alpha, beta, warmup, repeat)`, and a row key of experiment key plus `method_id`. Render complete managed content between these markers:

```python
BEGIN = "<!-- SGEMM_RESULTS_BEGIN -->"
END = "<!-- SGEMM_RESULTS_END -->"

def update_results_file(path: Path, new_records: Sequence[BenchmarkRecord]) -> None:
    old_text = path.read_text(encoding="utf-8") if path.exists() else ""
    old_records = parse_managed_records(old_text)
    rendered = render_results(upsert_records(old_records, new_records))
    prefix, suffix = split_managed_region(old_text)
    content = prefix + BEGIN + "\n" + rendered + "\n" + END + suffix
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
```

Embed each complete record as a compact HTML comment immediately before its displayed row so parsing never depends on rounded table text. Calculate `speedup_vs_naive` and `percent_cublas` only while rendering records in the same experiment group.

- [ ] **Step 4: Run result tests and verify GREEN**

Run: `python3 -m unittest tests.test_results -v`

Expected: all result tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add sgemm_tools/__init__.py sgemm_tools/results.py tests/test_results.py
git commit -m "feat: add benchmark result reporting"
```

### Task 2: CPU-testable benchmark contracts

**Files:**
- Create: `include/sgemm/benchmark.hpp`
- Create: `src/reference.cpp`
- Create: `tests/reference_test.cpp`
- Create: `CMakeLists.txt`

- [ ] **Step 1: Write the failing native reference test**

Test the exact `C = alpha*A*B + beta*C0` result for a 2x3 times 3x2 problem, deterministic input reproducibility, combined tolerance, and operation count:

```cpp
int main() {
  const sgemm::Problem p{2, 2, 3, 0.5F, 0.25F};
  const std::vector<float> a{1, 2, 3, 4, 5, 6};
  const std::vector<float> b{7, 8, 9, 10, 11, 12};
  const std::vector<float> c0{4, 8, 12, 16};
  const auto got = sgemm::cpu_sgemm(p, a, b, c0);
  const std::vector<float> want{30, 34, 72.5F, 81};
  assert(sgemm::compare_vectors(got, want, 1e-6, 1e-6).passed);
  assert(sgemm::operation_count(p) == 32.0);
}
```

- [ ] **Step 2: Run a direct host compilation and verify RED**

Run: `c++ -std=c++17 -Iinclude tests/reference_test.cpp src/reference.cpp -o /tmp/sgemm_reference_test`

Expected: compilation fails because the declarations/files are missing.

- [ ] **Step 3: Implement the contracts and CPU reference**

Declare `Problem`, `RunOptions`, `ErrorStats`, `BenchmarkResult`, `make_inputs`, `cpu_sgemm`, `compare_vectors`, and `operation_count`. Use checked `size_t` multiplication before allocating. The operation count must return:

```cpp
double operation_count(const Problem& p) {
  return 2.0 * p.m * p.n * p.k + 2.0 * p.m * p.n;
}
```

Add a CMake project using C++17/CUDA17 and a CPU-only `reference_test`; later tasks add the CUDA targets after their source files exist. When the caller does not set `CMAKE_CUDA_ARCHITECTURES`, use `75;80;86;89;90` so configuration does not require a working GPU driver, while still allowing an explicit override.

- [ ] **Step 4: Compile and run the native reference test**

Run: `c++ -std=c++17 -Iinclude tests/reference_test.cpp src/reference.cpp -o /tmp/sgemm_reference_test && /tmp/sgemm_reference_test`

Expected: exit code 0.

- [ ] **Step 5: Commit Task 2**

```bash
git add CMakeLists.txt include/sgemm/benchmark.hpp src/reference.cpp tests/reference_test.cpp
git commit -m "feat: define SGEMM benchmark contracts"
```

### Task 3: Kernel registry and Kernels 1–2

**Files:**
- Create: `include/sgemm/kernels.cuh`
- Create: `src/kernels/registry.cu`
- Create: `src/kernels/kernel_01_naive.cu`
- Create: `src/kernels/kernel_02_coalesced.cu`
- Create: `tests/registry_test.cu`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write a failing registry test**

Assert that IDs 1 and 2 resolve by both number and name, aliases are unique, and an unknown selector returns no value:

```cpp
int main() {
  const auto kernels = sgemm::phase_a_kernels();
  assert(kernels.size() == 2);
  assert(sgemm::find_kernel("1")->name == "naive");
  assert(sgemm::find_kernel("coalesced")->id == 2);
  assert(!sgemm::find_kernel("missing"));
}
```

- [ ] **Step 2: Compile the registry test and verify RED**

Run: `nvcc -std=c++17 -Iinclude tests/registry_test.cu src/kernels/registry.cu src/kernels/kernel_01_naive.cu src/kernels/kernel_02_coalesced.cu -o /tmp/sgemm_registry_test`

Expected: compilation fails because the registry and wrappers are absent.

- [ ] **Step 3: Implement the stable launch interface and Kernels 1–2**

Use this registry contract:

```cpp
using LaunchFn = cudaError_t (*)(const Problem&, const float*, const float*, float*, cudaStream_t);
struct KernelSpec { int id; std::string_view name; std::string_view description; LaunchFn launch; };
std::vector<KernelSpec> phase_a_kernels();
std::optional<KernelSpec> find_kernel(std::string_view selector);
```

Kernel 1 maps a 2D 32x32 block to one C element per thread. Kernel 2 uses a 1D 1024-thread block and maps `threadIdx.x / 32` to the output row and `% 32` to the consecutive output column. Both use grid order `x -> N columns`, `y -> M rows`, const/restrict input pointers, bounds guards, and return `cudaGetLastError()` from their wrapper.

- [ ] **Step 4: Compile and run the registry test**

Run: `nvcc -std=c++17 -Iinclude tests/registry_test.cu src/kernels/registry.cu src/kernels/kernel_01_naive.cu src/kernels/kernel_02_coalesced.cu -o /tmp/sgemm_registry_test && /tmp/sgemm_registry_test`

Expected: exit code 0; the test performs no device launch and therefore remains runnable without a driver.

- [ ] **Step 5: Commit Task 3**

```bash
git add CMakeLists.txt include/sgemm/kernels.cuh src/kernels/registry.cu src/kernels/kernel_01_naive.cu src/kernels/kernel_02_coalesced.cu tests/registry_test.cu
git commit -m "feat: add naive and coalesced SGEMM kernels"
```

### Task 4: Shared-memory and 1D block-tiling kernels

**Files:**
- Create: `src/kernels/kernel_03_shared.cu`
- Create: `src/kernels/kernel_04_blocktiling_1d.cu`
- Modify: `src/kernels/registry.cu`
- Modify: `CMakeLists.txt`
- Test: `tests/registry_test.cu`

- [ ] **Step 1: Extend the registry test and verify RED**

Expect IDs/names `(3, "shared")` and `(4, "blocktiling-1d")`, then compile the same direct `nvcc` command including the new filenames. It must fail because the wrappers are not defined.

- [ ] **Step 2: Implement Kernel 3 with boundary-safe tiles**

Use `BLOCK=32`, two `BLOCK*BLOCK` shared arrays, coalesced loads with zero fill, one synchronization after loads and one after the dot product. Compute `row = blockIdx.y*BLOCK+threadIdx.y` and `col = blockIdx.x*BLOCK+threadIdx.x`; advance tiles by `BLOCK` along K.

- [ ] **Step 3: Implement Kernel 4 and correct the legacy A-bound bug**

Use `BM=64, BN=64, BK=8, TM=8`, `BN*(BM/TM)=512` threads and `TM` accumulators per thread. Derive A load bounds from `blockIdx.y*BM + a_load_row`, not from the thread's `c_row_start + a_load_row`. Zero-fill A/B tile tails and guard every C store.

- [ ] **Step 4: Compile and run the extended registry test**

Run the direct `nvcc` registry test with sources 1–4.

Expected: four registered methods, exit code 0.

- [ ] **Step 5: Commit Task 4**

```bash
git add CMakeLists.txt src/kernels/registry.cu src/kernels/kernel_03_shared.cu src/kernels/kernel_04_blocktiling_1d.cu tests/registry_test.cu
git commit -m "feat: add shared and 1D tiled SGEMM kernels"
```

### Task 5: 2D block-tiling kernel

**Files:**
- Create: `src/kernels/kernel_05_blocktiling_2d.cu`
- Modify: `src/kernels/registry.cu`
- Modify: `CMakeLists.txt`
- Test: `tests/registry_test.cu`

- [ ] **Step 1: Add the Kernel 5 registry expectation and verify RED**

Add `assert(find_kernel("5")->name == "blocktiling-2d")` and expect five Phase-A kernels. Compile with the new source path and confirm the missing implementation fails.

- [ ] **Step 2: Implement a boundary-safe 2D tiled kernel**

Use `BM=64, BN=64, BK=8, TM=8, TN=8` and 64 threads. Each thread cooperatively loads `BM*BK + BK*BN` scalar values using a linear thread ID and strided loops, then computes a TMxTN register tile:

```cpp
for (int dot = 0; dot < BK; ++dot) {
  #pragma unroll
  for (int i = 0; i < TM; ++i) reg_m[i] = as[(thread_row*TM+i)*BK+dot];
  #pragma unroll
  for (int j = 0; j < TN; ++j) reg_n[j] = bs[dot*BN+thread_col*TN+j];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i*TN+j] += reg_m[i]*reg_n[j];
}
```

All cooperative loads zero-fill out-of-range rows/columns/K tails; all 64 stores are individually guarded.

- [ ] **Step 3: Compile and run the five-kernel registry test**

Run the direct `nvcc` registry command with all five kernel sources.

Expected: exit code 0.

- [ ] **Step 4: Commit Task 5**

```bash
git add CMakeLists.txt src/kernels/registry.cu src/kernels/kernel_05_blocktiling_2d.cu tests/registry_test.cu
git commit -m "feat: add 2D block tiled SGEMM kernel"
```

### Task 6: Native correctness and benchmark runner

**Files:**
- Create: `src/benchmark.cu`
- Create: `tests/test_native_cli.py`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing CLI contract tests**

Compile the executable in `setUpClass` when `nvcc` exists. Assert `--list` returns IDs 1–5 without opening a CUDA device, unknown kernels exit 2, invalid dimensions exit 2, and `--help` documents every option. GPU-executing assertions use `@unittest.skipUnless(gpu_available(), "CUDA driver/device unavailable")`.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `python3 -m unittest tests.test_native_cli -v`

Expected: build/import failure because `src/benchmark.cu` is missing.

- [ ] **Step 3: Implement parsing and CUDA RAII**

Support `--kernel`, `--all`, `--list`, `--m`, `--n`, `--k`, `--alpha`, `--beta`, `--warmup`, `--repeat`, `--seed`, `--atol`, `--rtol`, `--check`, `--check-only`, and `--json`. Defaults are `M=N=K=1024`, `alpha=0.8`, `beta=0.2`, five warmups, and twenty timed samples. Use move-only device-buffer and event wrappers. Validate positive sizes, nonnegative warmup, positive repeat, finite alpha/beta/tolerances, and mutually exclusive selector flags. When tolerances are not supplied, use `atol=1e-4*max(1,K)` and `rtol=1e-4`.

- [ ] **Step 4: Implement fair validation and timing**

Allocate both `d_C0` and `d_C`. For each selected method: restore C with device-to-device copy from `d_C0` before validation, launch once, synchronize, copy/compare; restore C again before every warmup/timed launch so nonzero beta always sees the same input; synchronize; record one CUDA-event sample per launch; report median and minimum. Continue after a method failure in `--all`, emit one JSON record per method, and return 1 if any method failed.

- [ ] **Step 5: Run CPU-visible CLI tests and conditional GPU checks**

Run: `python3 -m unittest tests.test_native_cli -v`

Expected on the current host: help/list/error tests pass; GPU execution tests are skipped with the driver-unavailable reason.

- [ ] **Step 6: Commit Task 6**

```bash
git add CMakeLists.txt src/benchmark.cu tests/test_native_cli.py
git commit -m "feat: add correctness checked CUDA benchmark runner"
```

### Task 7: Python build and orchestration CLI

**Files:**
- Create: `sgemm_tools/runner.py`
- Create: `benchmark.py`
- Create: `tests/test_runner.py`
- Create: `tests/test_cuda_integration.py`

- [ ] **Step 1: Write failing command-construction and parsing tests**

Assert `native_arguments` forwards every matrix/timing selector, `parse_json_lines` ignores human-readable lines but rejects malformed lines beginning with `{`, and `build_with_nvcc` includes all Phase-A sources and `-std=c++17` without shell interpolation.

- [ ] **Step 2: Run runner tests and verify RED**

Run: `python3 -m unittest tests.test_runner -v`

Expected: missing `sgemm_tools.runner` import.

- [ ] **Step 3: Implement dependency-free build and run helpers**

Use `subprocess.run(list[str], check=False, text=True, capture_output=True)`. Prefer an existing `build/sgemm_bench`; when building, use CMake if available and otherwise invoke `nvcc` directly into `build/sgemm_bench`. Never invoke through `shell=True`.

- [ ] **Step 4: Implement `benchmark.py`**

Mirror the native selectors, add `--build`, `--no-build`, `--build-dir`, `--update-results`, and `--results-file`. Stream the native human output, parse JSON records, update results only when requested, and propagate native/build failures.

- [ ] **Step 5: Add conditional end-to-end tests**

Compile whenever `nvcc` exists. Execute `--list` without a GPU. Execute `--all --check-only --m 37 --n 41 --k 29` only when a device probe succeeds, and assert five passing JSON records.

- [ ] **Step 6: Run all Python tests**

Run: `python3 -m unittest discover -s tests -v`

Expected on the current host: Python/native non-GPU tests pass and GPU integration is explicitly skipped.

- [ ] **Step 7: Commit Task 7**

```bash
git add benchmark.py sgemm_tools/runner.py tests/test_runner.py tests/test_cuda_integration.py
git commit -m "feat: automate SGEMM builds and batch runs"
```

### Task 8: Results seed, ignore policy, and README

**Files:**
- Create: `RESULTS.md`
- Create: `.gitignore`
- Create: `README.md`
- Test: `tests/test_docs.py`

- [ ] **Step 1: Write failing documentation consistency tests**

Assert every registered Phase-A method appears in README, every displayed command is supported by Python `--help`, `RESULTS.md` contains exactly one managed marker pair, and `.gitignore` covers `/build/`, `*.o`, `*.exe`, `*.exp`, `*.lib`, `__pycache__/`, `.pytest_cache/`, and `*.tmp` while not ignoring `RESULTS.md`.

- [ ] **Step 2: Run docs tests and verify RED**

Run: `python3 -m unittest tests.test_docs -v`

Expected: missing README/RESULTS/.gitignore assertions fail.

- [ ] **Step 3: Write the result seed and ignore rules**

Create an empty managed results section with reproduction guidance. Ignore future build products and caches. Do not run `git rm --cached` on already tracked legacy binaries in this task.

- [ ] **Step 4: Write the Chinese-first README**

Document project purpose, tutorial mapping for Kernels 1–5, prerequisites, CMake and Python-direct build paths, all single/batch/check/update commands, timing/FLOP formula, output interpretation, directory map, adding a kernel, current GPU-driver limitation, and troubleshooting.

- [ ] **Step 5: Run documentation tests**

Run: `python3 -m unittest tests.test_docs -v`

Expected: all documentation consistency tests pass.

- [ ] **Step 6: Commit Task 8**

```bash
git add .gitignore README.md RESULTS.md tests/test_docs.py
git commit -m "docs: document SGEMM benchmark workflow"
```

### Task 9: Phase-A verification and legacy handoff

**Files:**
- Modify only if verification exposes defects in Phase-A files.

- [ ] **Step 1: Run whitespace and placeholder checks**

Run: `git diff --check && rg -n 'TB[D]|TO[D]O|待[定]|占[位]' README.md RESULTS.md include src sgemm_tools tests`

Expected: `git diff --check` exits 0 and the placeholder scan has no matches.

- [ ] **Step 2: Run the complete CPU/Python suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: zero failures; GPU-only cases may be skipped with an explicit unavailable-driver message.

- [ ] **Step 3: Build both supported ways available on this host**

Run: `python3 benchmark.py --build --list`

Expected: direct-nvcc fallback builds and lists five methods. If CMake becomes available, also run `cmake -S . -B build/cmake -G Ninja && cmake --build build/cmake` and expect success.

- [ ] **Step 4: Probe real CUDA execution**

Run: `python3 benchmark.py --all --check-only --m 37 --n 41 --k 29`

Expected on a GPU host: five passing methods. Expected on the current host: nonzero exit with a clear CUDA-driver-unavailable diagnostic; report this as blocked runtime evidence, not a passing GPU test.

- [ ] **Step 5: Review requirements and repository state**

Confirm single selector, batch selector, C0 reset, correctness fields, JSON records, idempotent result updates, README, and ignore rules. Confirm the pre-existing modifications to `SGEMM.cu`, `script.py`, and `SGEMM.exe` remain preserved and unstaged unless explicitly requested.

- [ ] **Step 6: Commit verification-only fixes if required**

If verification required fixes, stage only the exact files shown as changed by `git status --short`, review `git diff --cached`, then commit them with `git commit -m "fix: address phase A verification findings"`. If no fixes were required, do not create an empty commit.
