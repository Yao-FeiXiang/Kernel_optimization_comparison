# CUDA SGEMM Phase B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the verified Phase-A framework with vectorized access, compile-time-candidate autotuning, warp tiling, and an optional cuBLAS baseline.

**Architecture:** Preserve the native registry and Python result pipeline while extending method metadata with preparation and implementation-detail fields. Optimized kernels use guarded scalar fallbacks for unsupported shapes, autotuning happens before timed measurement, and cuBLAS follows the same input/reset/timing contract as custom methods.

**Tech Stack:** CUDA C++17, CUDA Runtime API, cuBLAS when available, CMake 3.24+, Python 3.9+ standard library, `unittest`.

---

## File map

- Modify `include/sgemm/benchmark.hpp`: implementation detail and availability fields.
- Modify `include/sgemm/kernels.cuh`: preparation hook and complete-method registry.
- Create `src/kernels/kernel_06_vectorized.cu`: transposed shared-memory A and float4 path.
- Create `src/kernels/kernel_09_autotuned.cu`: precompiled candidate family and selection.
- Create `src/kernels/kernel_10_warptiling.cu`: warp-tiled kernel and scalar tail fallback.
- Create `src/kernels/cublas_baseline.cu`: optional method 0 wrapper.
- Modify `src/kernels/registry.cu`: methods 0, 6, 9, and 10.
- Modify `src/benchmark.cu`: prepare phase, unavailable status, method details and cuBLAS cleanup.
- Modify `CMakeLists.txt`: optional cuBLAS detection/link and new sources.
- Modify `sgemm_tools/results.py`: display variant/config and cuBLAS-relative metrics.
- Modify `sgemm_tools/runner.py`: new sources and optional `-lcublas` detection.
- Modify `benchmark.py`: `--tune` and complete-set selection behavior.
- Modify `tests/`: schema, registry, CLI, correctness, fallback and documentation coverage.
- Modify `README.md` and `RESULTS.md`: Phase-B method documentation and tables.

### Task 1: Extend result and registry contracts

**Files:**
- Modify: `include/sgemm/benchmark.hpp`
- Modify: `include/sgemm/kernels.cuh`
- Modify: `sgemm_tools/results.py`
- Test: `tests/test_results.py`
- Test: `tests/registry_test.cu`

- [ ] **Step 1: Write failing schema and registry-contract tests**

Add records containing `available`, `implementation_variant`, and `configuration`. Assert unavailable methods render as `unavailable` without numeric speedups, and cuBLAS-relative percentages are calculated only when a passing method-0 record exists in the same experiment group. Extend native compile-time assertions for this interface:

```cpp
struct DeviceOperands {
  const float* a;
  const float* b;
  const float* c0;
  float* c;
  cudaStream_t stream;
};
struct PrepareResult { bool ok; std::string configuration; std::string message; };
using PrepareFn = PrepareResult (*)(const Problem&, const DeviceOperands&);
using LaunchFn = LaunchResult (*)(const Problem&, const DeviceOperands&);
using CleanupFn = void (*)();
struct KernelSpec {
  int id;
  std::string_view name;
  std::string_view description;
  PrepareFn prepare;
  LaunchFn launch;
  CleanupFn cleanup;
};
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest tests.test_results -v` and compile `tests/registry_test.cu`.

Expected: failures for missing fields/types.

- [ ] **Step 3: Implement backward-compatible schema parsing and launch results**

Default missing old-record fields to `available=True` and empty strings. Convert existing wrappers from `cudaError_t` to `LaunchResult{ok, message, detail}` without changing their launch shapes. Use no-op prepare and cleanup hooks for fixed kernels. Treat prepare `configuration` and launch `detail` as separate JSON fields.

- [ ] **Step 4: Run Phase-A regression tests**

Run: `python3 -m unittest discover -s tests -v` and the direct `nvcc` registry compile.

Expected: zero failures; GPU cases may remain explicitly skipped.

- [ ] **Step 5: Commit Task 1**

```bash
git add include/sgemm/benchmark.hpp include/sgemm/kernels.cuh src/kernels/registry.cu src/kernels/kernel_0*.cu src/benchmark.cu sgemm_tools/results.py tests/test_results.py tests/registry_test.cu
git commit -m "refactor: support prepared benchmark methods"
```

### Task 2: Vectorized-memory method

**Files:**
- Create: `src/kernels/kernel_06_vectorized.cu`
- Modify: `src/kernels/registry.cu`
- Modify: `CMakeLists.txt`
- Modify: `sgemm_tools/runner.py`
- Test: `tests/registry_test.cu`
- Test: `tests/test_cuda_integration.py`

- [ ] **Step 1: Add failing method-6 and shape-path tests**

Expect `(6, "vectorized")` in the registry. On a GPU, test an aligned `128x128x64` shape and an unaligned `131x127x35` shape; assert both pass and their JSON details are respectively `float4` and `scalar-fallback`.

- [ ] **Step 2: Verify RED**

Run the registry compile and `python3 -m unittest tests.test_cuda_integration -v`.

Expected: missing method/source failure before GPU tests can pass.

- [ ] **Step 3: Implement Kernel 6**

Use `BM=128, BN=128, BK=8, TM=8, TN=8` with 256 threads. For inputs satisfying pointer 16-byte alignment plus `K%4==0` and `N%4==0`, load A/B as `float4`, transpose A into `As[k*BM+m]`, retain B as `Bs[k*BN+n]`, and store aligned C as `float4`. If any condition fails, launch the boundary-safe Kernel-5 scalar wrapper and return detail `scalar-fallback`.

- [ ] **Step 4: Run registry and conditional GPU tests**

Expected on a GPU host: both vector and fallback paths pass. On the current host: compilation/list tests pass and device tests skip explicitly.

- [ ] **Step 5: Commit Task 2**

```bash
git add CMakeLists.txt src/kernels/kernel_06_vectorized.cu src/kernels/registry.cu sgemm_tools/runner.py tests/registry_test.cu tests/test_cuda_integration.py
git commit -m "feat: add vectorized SGEMM kernel"
```

### Task 3: Precompiled-candidate autotuning

**Files:**
- Create: `src/kernels/kernel_09_autotuned.cu`
- Modify: `src/kernels/registry.cu`
- Modify: `src/benchmark.cu`
- Modify: `benchmark.py`
- Modify: `CMakeLists.txt`
- Modify: `sgemm_tools/runner.py`
- Test: `tests/test_native_cli.py`
- Test: `tests/test_cuda_integration.py`

- [ ] **Step 1: Write failing candidate-validation and CLI tests**

Expect method `(9, "autotuned")`; assert `--tune` selects method 9, and `--kernel autotuned --repeat 0` remains invalid. Unit-test candidate legality for threads per block, positive tile sizes, exact divisibility of BM/TM and BN/TN, and shared-memory size.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.test_native_cli tests.test_cuda_integration -v`

Expected: missing method and option failures.

- [ ] **Step 3: Implement templated candidate kernels and pre-timing selection**

Instantiate exactly these candidates:

```cpp
using CandidateList = TypeList<
  Tile<64, 64, 8, 8, 8>,
  Tile<128, 64, 8, 8, 8>,
  Tile<64, 128, 8, 8, 8>,
  Tile<128, 128, 8, 8, 8>,
  Tile<128, 128, 16, 8, 8>>;
```

The prepare hook restores C from `DeviceOperands.c0` with a device-to-device copy before every trial, warms each legal candidate once, times three trials, selects the lowest median, and caches by `(device, M, N, K)`. Candidate search time is excluded from the reported method latency. Return the selected `BMxBNxBK/TMxTN` string as `configuration`. Candidate kernels use the transposed-A/vectorized path for aligned shapes and boundary-safe scalar cooperative loads otherwise.

- [ ] **Step 4: Run CPU-visible and conditional GPU tests**

Assert method listing and CLI parsing without a driver. On a GPU, assert one legal configuration is reported and the selected kernel passes correctness for aligned and tail shapes.

- [ ] **Step 5: Commit Task 3**

```bash
git add CMakeLists.txt benchmark.py include/sgemm/kernels.cuh src/benchmark.cu src/kernels/kernel_09_autotuned.cu src/kernels/registry.cu sgemm_tools/runner.py tests/test_native_cli.py tests/test_cuda_integration.py
git commit -m "feat: add SGEMM autotuning candidates"
```

### Task 4: Warp-tiled method

**Files:**
- Create: `src/kernels/kernel_10_warptiling.cu`
- Modify: `src/kernels/registry.cu`
- Modify: `CMakeLists.txt`
- Modify: `sgemm_tools/runner.py`
- Test: `tests/registry_test.cu`
- Test: `tests/test_cuda_integration.py`

- [ ] **Step 1: Add failing method-10 and correctness tests**

Expect `(10, "warptiling")`. On a GPU test `256x256x64`, `130x129x17`, nonzero beta, and repeat execution from restored C0.

- [ ] **Step 2: Verify RED**

Run registry and CUDA integration tests; expect missing method/source failure.

- [ ] **Step 3: Implement warp tiling with a guarded tail path**

Use `BM=128, BN=128, BK=16, WM=64, WN=64, TM=8, TN=4, WMITER=2, WNITER=2`. Map each warp to a WMxWN region, subdivide into WMITER x WNITER thread tiles, transpose A in shared memory, and accumulate each thread's disjoint register tile. For shapes not divisible by BM/BN/BK or not 16-byte aligned, invoke the boundary-safe Kernel-5 path and report `scalar-fallback`; otherwise report `warp-tiled`.

- [ ] **Step 4: Compile and run conditional correctness tests**

Expected on a GPU: aligned and fallback tests pass. Expected on the current host: compile/list pass, device cases explicitly skip.

- [ ] **Step 5: Commit Task 4**

```bash
git add CMakeLists.txt src/kernels/kernel_10_warptiling.cu src/kernels/registry.cu sgemm_tools/runner.py tests/registry_test.cu tests/test_cuda_integration.py
git commit -m "feat: add warp tiled SGEMM kernel"
```

### Task 5: Optional cuBLAS baseline

**Files:**
- Create: `src/kernels/cublas_baseline.cu`
- Modify: `src/kernels/registry.cu`
- Modify: `src/benchmark.cu`
- Modify: `CMakeLists.txt`
- Modify: `sgemm_tools/runner.py`
- Test: `tests/registry_test.cu`
- Test: `tests/test_cuda_integration.py`

- [ ] **Step 1: Write failing available/unavailable baseline tests**

Expect method `(0, "cublas")` to always be listable. When built with cuBLAS, assert it is available and correct; when deliberately compiled with `SGEMM_ENABLE_CUBLAS=0`, assert it emits `status=unavailable`, continues other `--all` methods, and does not make the batch exit fail.

- [ ] **Step 2: Verify RED**

Run no-cuBLAS registry build plus integration tests and observe the missing method/status behavior.

- [ ] **Step 3: Implement row-major cuBLAS SGEMM**

Create one handle during prepare, bind the benchmark stream, and invoke column-major cuBLAS with swapped operands/dimensions:

```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            p.n, p.m, p.k, &p.alpha,
            b, p.n, a, p.k, &p.beta, c, p.n);
```

Map `cublasStatus_t` to readable `LaunchResult` messages and destroy the handle during runner cleanup. CMake uses `find_package(CUDAToolkit)` and links `CUDA::cublas` only when found; direct nvcc build probes for the header/library and defines `SGEMM_ENABLE_CUBLAS` accordingly.

- [ ] **Step 4: Run builds and conditional baseline correctness tests**

Compile with and without cuBLAS. On a GPU, compare method 0 to the CPU reference. Without a GPU, assert both binaries list cublas with the correct availability.

- [ ] **Step 5: Commit Task 5**

```bash
git add CMakeLists.txt src/benchmark.cu src/kernels/cublas_baseline.cu src/kernels/registry.cu sgemm_tools/runner.py tests/registry_test.cu tests/test_cuda_integration.py
git commit -m "feat: add optional cuBLAS benchmark baseline"
```

### Task 6: Full comparison output and documentation

**Files:**
- Modify: `benchmark.py`
- Modify: `sgemm_tools/results.py`
- Modify: `README.md`
- Modify: `RESULTS.md`
- Modify: `tests/test_results.py`
- Modify: `tests/test_docs.py`

- [ ] **Step 1: Write failing full-table and documentation tests**

Assert sorted rows `1,2,3,4,5,6,9,10,0`, display of variant/configuration, speedup vs Naive, percentage of cuBLAS, and README explanations for vector alignment fallback, tuning scope, warp tiling, cuBLAS optionality, and reproduction commands.

- [ ] **Step 2: Verify RED**

Run: `python3 -m unittest tests.test_results tests.test_docs -v`

Expected: missing columns/content failures.

- [ ] **Step 3: Implement complete comparison rendering**

Render columns `ID`, `Method`, `Variant/config`, `Status`, `Median ms`, `GFLOP/s`, `vs Naive`, `% cuBLAS`, `Max abs error`, and `Max rel error`. Preserve stored Phase-A records and render em dashes when a baseline is absent.

- [ ] **Step 4: Update README and result reproduction guidance**

Describe methods 6/9/10/0, `--tune`, full `--all`, optional library detection, safe tail fallback, why results vary by GPU/CUDA/clock state, and how to regenerate the table.

- [ ] **Step 5: Run docs/result tests and commit**

Run: `python3 -m unittest tests.test_results tests.test_docs -v`

Expected: all pass.

```bash
git add README.md RESULTS.md benchmark.py sgemm_tools/results.py tests/test_results.py tests/test_docs.py
git commit -m "docs: cover complete SGEMM optimization suite"
```

### Task 7: Phase-B verification

**Files:**
- Modify only when verification reveals an in-scope defect.

- [ ] **Step 1: Run static hygiene checks**

Run: `git diff --check && rg -n 'TB[D]|TO[D]O|待[定]|占[位]' README.md RESULTS.md include src sgemm_tools tests`

Expected: no whitespace errors or placeholders.

- [ ] **Step 2: Run the full automated suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: zero failures; GPU-only cases may explicitly skip if the driver remains unavailable.

- [ ] **Step 3: Build and list the complete suite**

Run: `python3 benchmark.py --build --list`

Expected: methods 1, 2, 3, 4, 5, 6, 9, 10, and 0 listed with accurate availability.

- [ ] **Step 4: Run correctness and benchmark commands when a GPU is available**

Run:

```bash
python3 benchmark.py --all --check-only --m 131 --n 127 --k 35
python3 benchmark.py --all --m 1024 --n 1024 --k 1024 --warmup 5 --repeat 20 --update-results
```

Expected on a GPU: all available methods pass and `RESULTS.md` updates once per method/configuration. On the current host: report the driver limitation and do not create synthetic benchmark rows.

- [ ] **Step 5: Verify idempotent result updating**

On a GPU host, repeat the same update command and assert the managed record count is unchanged. Always run the unit-level idempotency test even without a GPU.

- [ ] **Step 6: Review the full spec checklist**

Confirm every Phase-A and Phase-B completion criterion, explicitly distinguish compile/test evidence from GPU runtime evidence, and preserve unrelated legacy working-tree modifications.
