# A100 Tutorial-Fidelity Tuning Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task. Use `test-driven-development` for each behavior change and `verification-before-completion` before reporting success.

**Goal:** Make the local SGEMM stages faithfully reflect Simon Boehm's tutorial and bring the optimized A100/4096 results close to the tutorial implementation while retaining the project's unified runner, correctness checks, automatic result tables, and arbitrary-size support.

**Architecture:** Keep each numbered method behind the existing `KernelSpec` API. Add small, public, compile-time tutorial configuration records so exact teaching parameters can be tested without a GPU. For methods 5, 9, and 10, dispatch aligned tutorial-sized problems to vectorized, bounds-free fast kernels and route incompatible/tail shapes through the existing safe kernel. Preparation selects and caches method 9's candidate and method 10's device profile; launches remain allocation-free.

**Tech Stack:** CUDA C++17, CUDA Runtime API, cuBLAS baseline, CMake/CTest, Python `unittest`, existing `benchmark.py` result pipeline.

---

## Task 1: Add executable tutorial configuration contracts

**Files:**

- Create: `include/sgemm/tutorial_config.hpp`
- Create: `tests/tutorial_config_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_cuda_integration.py`

### Step 1: Write the failing configuration test

Create `tests/tutorial_config_test.cpp` with compile-time checks for:

```cpp
#include "sgemm/tutorial_config.hpp"

using namespace sgemm::tutorial;
static_assert(kKernel5Fast == TileConfig{128, 128, 8, 8, 8});
static_assert(kKernel9Candidates[0] == TileConfig{64, 64, 16, 4, 4});
static_assert(kKernel9Candidates[5] == TileConfig{128, 128, 16, 8, 8});
static_assert(kKernel10A100.block_m == 64);
static_assert(kKernel10A100.warp_m == 32);
static_assert(kKernel10A100.warp_n_iterations == 1);
static_assert(kKernel10Generic.block_m == 128);
static_assert(kKernel10Generic.warp_m == 64);

int main() {}
```

Add a CPU-only CMake target and CTest entry. Extend `tests/test_cuda_integration.py` with one aligned-shape test that runs methods 5, 9, and 10 at `128x128x64` and asserts successful correctness plus non-fallback implementation variants.

### Step 2: Run the tests to verify RED

Run:

```bash
cmake -S . -B build -DSGEMM_ENABLE_CUBLAS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 -m unittest tests.test_cuda_integration.CudaIntegrationTest.test_tutorial_aligned_paths_report_fast_variants -v
```

Expected: compilation fails because `sgemm/tutorial_config.hpp` and the new preparation/variant behavior do not exist yet.

### Step 3: Add the minimal configuration model

Create a standard C++ header containing comparable `TileConfig` and `WarpConfig` structs and these `inline constexpr` objects:

```cpp
inline constexpr TileConfig kKernel5Fast{128, 128, 8, 8, 8};
inline constexpr std::array<TileConfig, 6> kKernel9Candidates{{
    {64, 64, 16, 4, 4}, {64, 64, 8, 8, 8},
    {128, 64, 8, 8, 8}, {64, 128, 8, 8, 8},
    {128, 128, 8, 8, 8}, {128, 128, 16, 8, 8},
}};
inline constexpr WarpConfig kKernel10A100{/* BM=64, BN=128, BK=16,
                                           WM=32, WN=64, WNITER=1,
                                           TM=4, TN=4, threads=128 */};
inline constexpr WarpConfig kKernel10Generic{/* tutorial A6000 values */};
```

Only make the CPU configuration target pass in this task; the GPU integration assertion remains the RED contract for later tasks.

### Step 4: Run the CPU test to verify GREEN

Run:

```bash
cmake --build build -j --target tutorial_config_test
ctest --test-dir build -R tutorial_config_test --output-on-failure
python3 -m unittest tests.test_runner -v
```

Expected: the new configuration target and existing runner tests pass.

### Step 5: Commit

```bash
git add include/sgemm/tutorial_config.hpp tests/tutorial_config_test.cpp tests/test_cuda_integration.py CMakeLists.txt
git commit -m "test: define tutorial kernel configurations"
```

## Task 2: Restore the tutorial's naive/coalesced teaching contrast

**Files:**

- Modify: `src/kernels/kernel_01_naive.cu`
- Modify: `tests/test_cuda_integration.py`

### Step 1: Write the failing behavior assertion

Add a GPU integration test that runs methods 1 and 2 on a large aligned square, checks both results, and records that method 2 is faster than method 1 by a conservative factor. Keep the ratio assertion in a separately named performance test so ordinary correctness runs can remain stable:

```python
self.assertGreater(coalesced.gflops, naive.gflops * 1.5)
```

Use `m=n=k=1024`, `warmup=3`, and `repeat=20` for this focused contract.

### Step 2: Run it to verify RED

Run the focused integration test on GPU0.

Expected: it fails because the current method 1 and method 2 both map `threadIdx.x` to contiguous output columns and perform similarly.

### Step 3: Restore method 1's deliberately uncoalesced mapping

Change only the thread/output mapping in method 1:

```cpp
const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
const int column = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
```

Keep bounds checks, `alpha`/`beta`, and all public identifiers unchanged. Method 2 remains the explicitly coalesced version.

### Step 4: Verify correctness and contrast

Run:

```bash
python3 -m unittest tests.test_cuda_integration.CudaIntegrationTest.test_all_kernels_match_cpu_reference_on_tail_shape -v
python3 -m unittest tests.test_cuda_integration.CudaIntegrationTest.test_coalescing_stage_improves_large_square -v
```

Expected: both pass; method 2 is materially faster than method 1.

### Step 5: Commit

```bash
git add src/kernels/kernel_01_naive.cu tests/test_cuda_integration.py
git commit -m "fix: restore naive memory access lesson"
```

## Task 3: Add method 5's 128x128 aligned fast path

**Files:**

- Modify: `src/kernels/kernel_05_blocktiling_2d.cu`
- Modify: `tests/test_cuda_integration.py`

### Step 1: Tighten the failing aligned-path test

Assert method 5 reports `"implementation_variant":"register-tile-2d-fast"` for `128x128x64`, and `"implementation_variant":"tail-safe"` for `131x127x35`.

Run it and confirm the current single 64x64 kernel reports only `register-tile-2d`.

### Step 2: Implement the tutorial fast kernel

Retain the current 64x64 bounds-checked kernel as the safe fallback. Add a templated or dedicated bounds-free fast kernel using:

- `BM=128`, `BN=128`, `BK=8`, `TM=8`, `TN=8`
- 256 one-dimensional threads
- vectorized/coalesced shared-memory loads
- `__launch_bounds__(256)`
- no per-element bounds branches inside the fast kernel

Dispatch to the fast path only when dimensions and base pointers satisfy its tile/vector alignment. Return the exact variant strings in the launch detail.

### Step 3: Verify RED becomes GREEN

Run the focused method 5 variant test, the tail-shape all-method test, and the aligned all-method correctness test.

Expected: both variants are selected as specified and all numerical checks pass.

### Step 4: Measure the isolated stage

Run methods 4 and 5 at 4096 with `warmup=5`, `repeat=50` on GPU0. Confirm method 5 is at least 1.1x method 4 under an idle device; if not, inspect compiler register/shared-memory reports before changing tile values away from the tutorial contract.

### Step 5: Commit

```bash
git add src/kernels/kernel_05_blocktiling_2d.cu tests/test_cuda_integration.py
git commit -m "perf: add tutorial 2D block tiling fast path"
```

## Task 4: Rebuild method 9 as vectorized candidate autotuning

**Files:**

- Modify: `src/kernels/kernel_09_autotuned.cu`
- Modify: `tests/test_cuda_integration.py`

### Step 1: Add failing autotuner contracts

For an aligned `128x128x64` problem assert:

- preparation succeeds;
- configuration is one of the six names derived from `kKernel9Candidates`;
- implementation variant is `autotuned-vectorized`.

For `131x127x35`, assert variant `tail-safe` and no candidate timing is attempted.

### Step 2: Implement the vectorized candidate family

Replace the scalar, bounds-checked candidate kernel with a template that:

- uses `float4` global loads/stores where the selected tile permits;
- transposes A into shared memory to avoid bank conflicts;
- has compile-time BM/BN/BK/TM/TN and matching `__launch_bounds__`;
- omits bounds checks because selection requires compatible dimensions;
- writes `alpha * AB + beta * C` exactly.

Instantiate all six configurations from `tutorial_config.hpp`, including the A100 `64x64x16, TM=TN=4` candidate.

### Step 3: Preserve stable autotuning semantics

For compatible shapes, time each candidate with one warmup and three event-timed samples, select the median, and cache by `(device, m, n, k)`. For incompatible shapes, cache a sentinel selecting `launch_blocktiling_2d` and return a clear `tail-safe` configuration. Ensure autotuning restores `C` from `C0` between every candidate/sample.

### Step 4: Verify correctness, cache, and performance

Run focused method 9 aligned/tail tests twice in one process to cover the cache, then all-method tail correctness. Benchmark methods 6 and 9 at 4096 with `warmup=5`, `repeat=50`; method 9 should reach at least 90% of method 6 and select a tutorial candidate.

### Step 5: Commit

```bash
git add src/kernels/kernel_09_autotuned.cu tests/test_cuda_integration.py
git commit -m "perf: vectorize tutorial autotuning candidates"
```

## Task 5: Make method 10 select the exact A100 warp-tiled profile

**Files:**

- Modify: `include/sgemm/kernels.cuh`
- Modify: `src/kernels/kernel_10_warptiling.cu`
- Modify: `src/kernels/registry.cu`
- Modify: `tests/registry_test.cu`
- Modify: `tests/test_cuda_integration.py`

### Step 1: Write failing preparation/variant assertions

Add `prepare_warptiling` to the public kernel API and registry test. On an A100, the integration test must assert:

```text
configuration: BM64_BN128_BK16_WM32_WN64_WNITER1_TM4_TN4_THREADS128
implementation_variant: warp-tiled-a100
```

Tail shapes must report `tail-safe`.

### Step 2: Generalize the warp kernel template

Template the current kernel on BM, BN, BK, WM, WN, WNITER, TM, and TN. Preserve vectorized loads, transposed A, vectorized C stores, and add `__launch_bounds__(128)`. Instantiate exactly:

- A100/SM80: `BM64 BN128 BK16 WM32 WN64 WNITER1 TM4 TN4 threads128`
- generic/A6000: `BM128 BN128 BK16 WM64 WN64 WNITER4 TM8 TN4 threads128`

Compute derived warp iteration values at compile time with static assertions covering thread and tile divisibility.

### Step 3: Select once during preparation

In `prepare_warptiling`, query the current device properties and select the A100 profile for compute capability 8.0, otherwise the generic tutorial profile. Cache the device/profile and return its configuration. `launch_warptiling` must use the prepared profile and route incompatible dimensions/pointers through method 5's safe dispatch.

### Step 4: Verify correctness and A100 performance

Run registry, aligned/tail integration, and all-method correctness tests. On GPU0, benchmark method 10 and cuBLAS at 4096 with `warmup=5`, `repeat=50`. Accept when method 10 is at least 85% of local cuBLAS and within 10% of the previously measured upstream tutorial kernel (~15.99 TFLOP/s), provided GPU utilization sampling shows no competing compute load.

### Step 5: Commit

```bash
git add include/sgemm/kernels.cuh src/kernels/kernel_10_warptiling.cu src/kernels/registry.cu tests/registry_test.cu tests/test_cuda_integration.py
git commit -m "perf: select tutorial warp tiling by GPU"
```

## Task 6: Integrate reproducible A100/4096 benchmarking and documentation

**Files:**

- Modify: `README.md`
- Modify: `RESULTS.md`
- Modify if needed: `benchmark.py`
- Modify if needed: `sgemm_tools/results.py`
- Modify: `.gitignore`

### Step 1: Write documentation/result regression tests first

Extend the existing result-generation tests to assert that updating a 4096 run:

- creates/updates a distinct A100/4096 benchmark group;
- preserves the existing A100/1024 group;
- renders the visual ranking and detailed Markdown table;
- records warmup/repeat, GPU, dimensions, configuration, and implementation variant.

Run the focused result tests and confirm RED for any missing grouping/metadata behavior before changing the generator.

### Step 2: Make only necessary pipeline changes

If the current result pipeline already satisfies the contract, do not refactor it. Otherwise, make the smallest change in `sgemm_tools/results.py`/`benchmark.py` to preserve multiple dimension groups and their visual tables. Keep generated build artifacts, temporary tuning output, and worktrees ignored in `.gitignore`.

### Step 3: Document tutorial-comparable usage

Update README with:

- why the tutorial table cannot be compared to the former 1024 run directly;
- the recommended A100 command using GPU0, `4096^3`, `warmup=5`, `repeat=50`, correctness checking, and automatic `RESULTS.md` update;
- single-method and batch-comparison commands;
- how the aligned fast path and arbitrary-size fallback differ;
- how to recognize and avoid a busy shared GPU.

### Step 4: Run the final benchmark and update results

First inspect GPU0 memory/utilization and running processes. When compute utilization is idle, run the project's recommended batch command on GPU0 and allow it to update `RESULTS.md` automatically. Do not overwrite the 1024 historical group.

### Step 5: Full verification

Run:

```bash
python3 -m unittest discover -s tests -v
cmake -S . -B build -DSGEMM_ENABLE_CUBLAS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
CUDA_VISIBLE_DEVICES=0 ./build/sgemm_bench --all --m 37 --n 41 --k 29 --check-only --json
CUDA_VISIBLE_DEVICES=0 ./build/sgemm_bench --all --m 128 --n 128 --k 64 --check-only --json
git diff --check
```

Inspect `RESULTS.md` visually and verify all nine methods are present or explicitly marked unavailable, the 4096 group is current, the 1024 group remains, and method 10 meets the documented acceptance conditions.

### Step 6: Commit

```bash
git add README.md RESULTS.md .gitignore benchmark.py sgemm_tools/results.py tests
git commit -m "docs: publish tutorial-comparable A100 results"
```

## Task 7: Final review and branch integration

### Step 1: Review the full diff

Compare the feature branch against its base, check for accidental generated binaries or unrelated changes, and verify every implementation variant/configuration string is consistent between code, tests, README, and RESULTS.

### Step 2: Re-run the completion gate

Run the complete verification commands from Task 6 using fresh output. Record the exact passing counts and final A100/4096 method 10/cuBLAS numbers.

### Step 3: Integrate automatically

Use the repository's normal non-destructive Git flow to merge the verified feature branch into `main`. Do not ask the user to choose among merge options; do not push unless a configured remote and the user's existing workflow explicitly require it.
