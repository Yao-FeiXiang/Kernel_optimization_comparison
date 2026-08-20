#include "sgemm/kernels.cuh"
#include "sgemm/tutorial_config.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>

namespace sgemm {
namespace {

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    tuned_vectorized_kernel(Problem problem,
                            const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c) {
  static_assert(BM % TM == 0 && BN % TN == 0);
  static_assert(BK % 4 == 0 && TN % 4 == 0);
  constexpr int threads = (BM / TM) * (BN / TN);
  static_assert(threads <= 1024);

  __shared__ float tile_a[BK * BM];
  __shared__ float tile_b[BK * BN];

  const int thread = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int thread_row = static_cast<int>(threadIdx.y);
  const int thread_column = static_cast<int>(threadIdx.x);
  const int block_row = static_cast<int>(blockIdx.y) * BM;
  const int block_column = static_cast<int>(blockIdx.x) * BN;
  float results[TM * TN] = {0.0F};

  for (int tile_start = 0; tile_start < problem.k; tile_start += BK) {
    for (int vector_index = thread; vector_index < BM * BK / 4;
         vector_index += threads) {
      const int float_index = vector_index * 4;
      const int local_row = float_index / BK;
      const int local_column = float_index % BK;
      const float4 loaded = *reinterpret_cast<const float4*>(
          a + static_cast<std::size_t>(block_row + local_row) * problem.k +
          tile_start + local_column);
      tile_a[(local_column + 0) * BM + local_row] = loaded.x;
      tile_a[(local_column + 1) * BM + local_row] = loaded.y;
      tile_a[(local_column + 2) * BM + local_row] = loaded.z;
      tile_a[(local_column + 3) * BM + local_row] = loaded.w;
    }
    for (int vector_index = thread; vector_index < BK * BN / 4;
         vector_index += threads) {
      const int float_index = vector_index * 4;
      const int local_row = float_index / BN;
      const int local_column = float_index % BN;
      *reinterpret_cast<float4*>(tile_b + float_index) =
          *reinterpret_cast<const float4*>(
              b + static_cast<std::size_t>(tile_start + local_row) * problem.n +
              block_column + local_column);
    }
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < BK; ++inner) {
      float register_m[TM];
      float register_n[TN];
#pragma unroll
      for (int row = 0; row < TM; ++row) {
        register_m[row] = tile_a[inner * BM + thread_row * TM + row];
      }
#pragma unroll
      for (int column = 0; column < TN; ++column) {
        register_n[column] = tile_b[inner * BN + thread_column * TN + column];
      }
#pragma unroll
      for (int row = 0; row < TM; ++row) {
#pragma unroll
        for (int column = 0; column < TN; ++column) {
          results[row * TN + column] += register_m[row] * register_n[column];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int row_offset = 0; row_offset < TM; ++row_offset) {
#pragma unroll
    for (int column_offset = 0; column_offset < TN; column_offset += 4) {
      const int row = block_row + thread_row * TM + row_offset;
      const int column = block_column + thread_column * TN + column_offset;
      const auto index = static_cast<std::size_t>(row) * problem.n + column;
      const float4 old = *reinterpret_cast<const float4*>(c + index);
      const int result_index = row_offset * TN + column_offset;
      const float4 output = make_float4(
          problem.alpha * results[result_index + 0] + problem.beta * old.x,
          problem.alpha * results[result_index + 1] + problem.beta * old.y,
          problem.alpha * results[result_index + 2] + problem.beta * old.z,
          problem.alpha * results[result_index + 3] + problem.beta * old.w);
      *reinterpret_cast<float4*>(c + index) = output;
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
LaunchResult launch_candidate(const Problem& problem, const DeviceOperands& operands) {
  const dim3 block(BN / TN, BM / TM);
  const dim3 grid(static_cast<unsigned int>(problem.n / BN),
                  static_cast<unsigned int>(problem.m / BM));
  tuned_vectorized_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, operands.stream>>>(
      problem, operands.a, operands.b, operands.c);
  return cuda_launch_result(cudaGetLastError(), "autotuned-vectorized");
}

struct Candidate {
  tutorial::TileConfig tile;
  const char* configuration;
  LaunchFn launch;
};

LaunchResult launch_kernel6_baseline(const Problem& problem,
                                     const DeviceOperands& operands) {
  LaunchResult result = launch_vectorized(problem, operands);
  if (result.ok && result.detail == "float4") {
    result.detail = "autotuned-vectorized";
  }
  return result;
}

constexpr auto kTiles = tutorial::kKernel9Candidates;
const std::array<Candidate, 6> kCandidates{{
    {kTiles[0], "BM64_BN64_BK16_TM4_TN4", launch_candidate<64, 64, 16, 4, 4>},
    {kTiles[1], "BM64_BN64_BK8_TM8_TN8", launch_candidate<64, 64, 8, 8, 8>},
    {kTiles[2], "BM128_BN64_BK8_TM8_TN8", launch_candidate<128, 64, 8, 8, 8>},
    {kTiles[3], "BM64_BN128_BK8_TM8_TN8", launch_candidate<64, 128, 8, 8, 8>},
    {kTiles[4], "BM128_BN128_BK8_TM8_TN8", launch_kernel6_baseline},
    {kTiles[5],
     "BM128_BN128_BK16_TM8_TN8",
     launch_candidate<128, 128, 16, 8, 8>},
}};

struct TuneCache {
  int device{-1};
  int m{-1};
  int n{-1};
  int k{-1};
  int selected{-1};
};

constexpr int kTailSafeSelection = -2;
TuneCache g_cache;

bool aligned16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

bool compatible(const Candidate& candidate,
                const Problem& problem,
                const DeviceOperands& operands) {
  return problem.m % candidate.tile.block_m == 0 &&
         problem.n % candidate.tile.block_n == 0 &&
         problem.k % candidate.tile.block_k == 0 && aligned16(operands.a) &&
         aligned16(operands.b) && aligned16(operands.c);
}

PrepareResult cuda_prepare_error(cudaError_t status, const char* operation) {
  return {false, true, {}, std::string(operation) + ": " + cudaGetErrorString(status)};
}

bool cache_matches(int device, const Problem& problem) {
  return g_cache.device == device && g_cache.m == problem.m && g_cache.n == problem.n &&
         g_cache.k == problem.k;
}

}  // namespace

PrepareResult prepare_autotuned(const Problem& problem, const DeviceOperands& operands) {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return cuda_prepare_error(status, "cudaGetDevice");
  }
  if (cache_matches(device, problem)) {
    if (g_cache.selected == kTailSafeSelection) {
      return {true, true, "tail-safe", {}};
    }
    if (g_cache.selected >= 0 &&
        g_cache.selected < static_cast<int>(kCandidates.size())) {
      return {true,
              true,
              kCandidates[static_cast<std::size_t>(g_cache.selected)].configuration,
              {}};
    }
  }

  const bool any_compatible = std::any_of(
      kCandidates.begin(), kCandidates.end(), [&](const Candidate& candidate) {
        return compatible(candidate, problem, operands);
      });
  if (!any_compatible) {
    g_cache = {device, problem.m, problem.n, problem.k, kTailSafeSelection};
    return {true, true, "tail-safe", {}};
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  status = cudaEventCreate(&start);
  if (status != cudaSuccess) {
    return cuda_prepare_error(status, "cudaEventCreate(start)");
  }
  status = cudaEventCreate(&stop);
  if (status != cudaSuccess) {
    cudaEventDestroy(start);
    return cuda_prepare_error(status, "cudaEventCreate(stop)");
  }

  const std::size_t c_bytes =
      static_cast<std::size_t>(problem.m) * problem.n * sizeof(float);
  float best_time = std::numeric_limits<float>::infinity();
  int best_index = -1;
  for (std::size_t candidate_index = 0; candidate_index < kCandidates.size();
       ++candidate_index) {
    const Candidate& candidate = kCandidates[candidate_index];
    if (!compatible(candidate, problem, operands)) {
      continue;
    }

    std::array<float, 3> samples{};
    status = cudaMemcpyAsync(
        operands.c, operands.c0, c_bytes, cudaMemcpyDeviceToDevice, operands.stream);
    const LaunchResult warmup = candidate.launch(problem, operands);
    if (status != cudaSuccess || !warmup.ok) {
      continue;
    }
    status = cudaStreamSynchronize(operands.stream);
    if (status != cudaSuccess) {
      continue;
    }

    bool valid = true;
    for (float& sample : samples) {
      status = cudaMemcpyAsync(
          operands.c, operands.c0, c_bytes, cudaMemcpyDeviceToDevice, operands.stream);
      if (status == cudaSuccess) {
        status = cudaEventRecord(start, operands.stream);
      }
      const LaunchResult launched = candidate.launch(problem, operands);
      if (status == cudaSuccess && launched.ok) {
        status = cudaEventRecord(stop, operands.stream);
      }
      if (status == cudaSuccess) {
        status = cudaEventSynchronize(stop);
      }
      if (status == cudaSuccess) {
        status = cudaEventElapsedTime(&sample, start, stop);
      }
      if (status != cudaSuccess || !launched.ok) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      continue;
    }
    std::sort(samples.begin(), samples.end());
    if (samples[1] < best_time) {
      best_time = samples[1];
      best_index = static_cast<int>(candidate_index);
    }
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (best_index < 0) {
    return {false, true, {}, "no autotuning candidate launched successfully"};
  }
  g_cache = {device, problem.m, problem.n, problem.k, best_index};
  return {true,
          true,
          kCandidates[static_cast<std::size_t>(best_index)].configuration,
          {}};
}

LaunchResult launch_autotuned(const Problem& problem, const DeviceOperands& operands) {
  if (g_cache.selected == kTailSafeSelection) {
    LaunchResult fallback = launch_blocktiling_2d(problem, operands);
    fallback.detail = "tail-safe";
    return fallback;
  }
  if (g_cache.selected < 0 || g_cache.selected >= static_cast<int>(kCandidates.size())) {
    return {false, "autotuned method was launched before preparation", {}};
  }
  return kCandidates[static_cast<std::size_t>(g_cache.selected)].launch(problem, operands);
}

}  // namespace sgemm
