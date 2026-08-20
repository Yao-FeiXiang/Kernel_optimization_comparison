#include "sgemm/kernels.cuh"
#include "sgemm/tutorial_config.hpp"

#include <cstdint>
#include <string>

namespace sgemm {
namespace {

constexpr int kWarpSize = 32;

template <int BM,
          int BN,
          int BK,
          int WM,
          int WN,
          int WNITER,
          int TM,
          int TN,
          int THREADS>
__global__ void __launch_bounds__(THREADS)
    warptiling_kernel(Problem problem,
                      const float* __restrict__ a,
                      const float* __restrict__ b,
                      float* __restrict__ c) {
  static_assert(THREADS == (BM / WM) * (BN / WN) * kWarpSize);
  static_assert((WM * WN) % (kWarpSize * TM * TN * WNITER) == 0);
  constexpr int WMITER = (WM * WN) / (kWarpSize * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;
  static_assert(WMITER > 0 && WM % WMITER == 0 && WN % WNITER == 0);
  static_assert((WSUBM / TM) * (WSUBN / TN) == kWarpSize);
  static_assert(BK % 4 == 0 && BN % 4 == 0 && TN % 4 == 0);
  constexpr int row_stride_a = (THREADS * 4) / BK;
  constexpr int row_stride_b = THREADS / (BN / 4);
  static_assert(row_stride_a > 0 && BM % row_stride_a == 0);
  static_assert(row_stride_b > 0 && BK % row_stride_b == 0);

  __shared__ float tile_a[BM * BK];
  __shared__ float tile_b[BK * BN];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpSize;
  const int warp_column = warp % (BN / WN);
  const int warp_row = warp / (BN / WN);
  const int lane = thread % kWarpSize;
  const int thread_column = lane % (WSUBN / TN);
  const int thread_row = lane / (WSUBN / TN);
  const int block_row = static_cast<int>(blockIdx.y) * BM;
  const int block_column = static_cast<int>(blockIdx.x) * BN;

  const int inner_row_a = thread / (BK / 4);
  const int inner_column_a = thread % (BK / 4);
  const int inner_row_b = thread / (BN / 4);
  const int inner_column_b = thread % (BN / 4);

  const float* block_a = a + static_cast<std::size_t>(block_row) * problem.k;
  const float* block_b = b + block_column;
  float* warp_c = c + static_cast<std::size_t>(block_row + warp_row * WM) * problem.n +
                  block_column + warp_column * WN;

  float results[WMITER * TM * WNITER * TN] = {0.0F};
  float register_m[WMITER * TM] = {0.0F};
  float register_n[WNITER * TN] = {0.0F};

  for (int tile_start = 0; tile_start < problem.k; tile_start += BK) {
#pragma unroll
    for (int offset = 0; offset < BM; offset += row_stride_a) {
      const float4 loaded = *reinterpret_cast<const float4*>(
          block_a + static_cast<std::size_t>(inner_row_a + offset) * problem.k +
          inner_column_a * 4);
      tile_a[(inner_column_a * 4 + 0) * BM + inner_row_a + offset] = loaded.x;
      tile_a[(inner_column_a * 4 + 1) * BM + inner_row_a + offset] = loaded.y;
      tile_a[(inner_column_a * 4 + 2) * BM + inner_row_a + offset] = loaded.z;
      tile_a[(inner_column_a * 4 + 3) * BM + inner_row_a + offset] = loaded.w;
    }
#pragma unroll
    for (int offset = 0; offset < BK; offset += row_stride_b) {
      *reinterpret_cast<float4*>(
          tile_b + (inner_row_b + offset) * BN + inner_column_b * 4) =
          *reinterpret_cast<const float4*>(
              block_b + static_cast<std::size_t>(inner_row_b + offset) * problem.n +
              inner_column_b * 4);
    }
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < BK; ++inner) {
#pragma unroll
      for (int warp_m = 0; warp_m < WMITER; ++warp_m) {
#pragma unroll
        for (int row = 0; row < TM; ++row) {
          register_m[warp_m * TM + row] =
              tile_a[inner * BM + warp_row * WM + warp_m * WSUBM +
                     thread_row * TM + row];
        }
      }
#pragma unroll
      for (int warp_n = 0; warp_n < WNITER; ++warp_n) {
#pragma unroll
        for (int column = 0; column < TN; ++column) {
          register_n[warp_n * TN + column] =
              tile_b[inner * BN + warp_column * WN + warp_n * WSUBN +
                     thread_column * TN + column];
        }
      }
#pragma unroll
      for (int warp_m = 0; warp_m < WMITER; ++warp_m) {
#pragma unroll
        for (int warp_n = 0; warp_n < WNITER; ++warp_n) {
#pragma unroll
          for (int row = 0; row < TM; ++row) {
#pragma unroll
            for (int column = 0; column < TN; ++column) {
              const int result_index =
                  (warp_m * TM + row) * (WNITER * TN) + warp_n * TN + column;
              results[result_index] += register_m[warp_m * TM + row] *
                                       register_n[warp_n * TN + column];
            }
          }
        }
      }
    }

    block_a += BK;
    block_b += static_cast<std::size_t>(BK) * problem.n;
    __syncthreads();
  }

#pragma unroll
  for (int warp_m = 0; warp_m < WMITER; ++warp_m) {
#pragma unroll
    for (int warp_n = 0; warp_n < WNITER; ++warp_n) {
      float* subtile_c =
          warp_c + static_cast<std::size_t>(warp_m * WSUBM) * problem.n +
          warp_n * WSUBN;
#pragma unroll
      for (int row = 0; row < TM; ++row) {
#pragma unroll
        for (int column = 0; column < TN; column += 4) {
          const auto index =
              static_cast<std::size_t>(thread_row * TM + row) * problem.n +
              thread_column * TN + column;
          const float4 old = *reinterpret_cast<const float4*>(subtile_c + index);
          const int result_index =
              (warp_m * TM + row) * (WNITER * TN) + warp_n * TN + column;
          const float4 output = make_float4(
              problem.alpha * results[result_index + 0] + problem.beta * old.x,
              problem.alpha * results[result_index + 1] + problem.beta * old.y,
              problem.alpha * results[result_index + 2] + problem.beta * old.z,
              problem.alpha * results[result_index + 3] + problem.beta * old.w);
          *reinterpret_cast<float4*>(subtile_c + index) = output;
        }
      }
    }
  }
}

enum class WarpProfile { kUnprepared, kA100, kGeneric };

struct WarpCache {
  int device{-1};
  WarpProfile profile{WarpProfile::kUnprepared};
};

WarpCache g_cache;

constexpr const char* kA100Configuration =
    "BM64_BN128_BK16_WM32_WN64_WNITER1_TM4_TN4_THREADS128";
constexpr const char* kGenericConfiguration =
    "BM128_BN128_BK16_WM64_WN64_WNITER4_TM8_TN4_THREADS128";

bool aligned16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

template <int BM,
          int BN,
          int BK,
          int WM,
          int WN,
          int WNITER,
          int TM,
          int TN,
          int THREADS>
LaunchResult launch_profile(const Problem& problem,
                            const DeviceOperands& operands,
                            const char* variant) {
  const bool compatible = problem.m % BM == 0 && problem.n % BN == 0 &&
                          problem.k % BK == 0 && aligned16(operands.a) &&
                          aligned16(operands.b) && aligned16(operands.c);
  if (!compatible) {
    LaunchResult fallback = launch_blocktiling_2d(problem, operands);
    fallback.detail = "tail-safe";
    return fallback;
  }

  const dim3 block(THREADS);
  const dim3 grid(static_cast<unsigned int>(problem.n / BN),
                  static_cast<unsigned int>(problem.m / BM));
  warptiling_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, THREADS>
      <<<grid, block, 0, operands.stream>>>(problem, operands.a, operands.b, operands.c);
  return cuda_launch_result(cudaGetLastError(), variant);
}

PrepareResult cuda_prepare_error(cudaError_t status, const char* operation) {
  return {false, true, {}, std::string(operation) + ": " + cudaGetErrorString(status)};
}

}  // namespace

PrepareResult prepare_warptiling(const Problem&, const DeviceOperands&) {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return cuda_prepare_error(status, "cudaGetDevice");
  }
  if (g_cache.device != device || g_cache.profile == WarpProfile::kUnprepared) {
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
      return cuda_prepare_error(status, "cudaGetDeviceProperties");
    }
    g_cache = {device,
               properties.major == 8 && properties.minor == 0 ? WarpProfile::kA100
                                                               : WarpProfile::kGeneric};
  }
  return {true,
          true,
          g_cache.profile == WarpProfile::kA100 ? kA100Configuration
                                                : kGenericConfiguration,
          {}};
}

LaunchResult launch_warptiling(const Problem& problem, const DeviceOperands& operands) {
  if (g_cache.profile == WarpProfile::kA100) {
    constexpr auto config = tutorial::kKernel10A100;
    return launch_profile<config.block_m,
                          config.block_n,
                          config.block_k,
                          config.warp_m,
                          config.warp_n,
                          config.warp_n_iterations,
                          config.thread_m,
                          config.thread_n,
                          config.threads>(problem, operands, "warp-tiled-a100");
  }
  if (g_cache.profile == WarpProfile::kGeneric) {
    constexpr auto config = tutorial::kKernel10Generic;
    return launch_profile<config.block_m,
                          config.block_n,
                          config.block_k,
                          config.warp_m,
                          config.warp_n,
                          config.warp_n_iterations,
                          config.thread_m,
                          config.thread_n,
                          config.threads>(problem, operands, "warp-tiled-generic");
  }
  return {false, "warptiling method was launched before preparation", {}};
}

}  // namespace sgemm
