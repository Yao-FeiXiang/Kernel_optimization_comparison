#include "sgemm/kernels.cuh"

#include <cstdint>

namespace sgemm {
namespace {

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 16;
constexpr int kWarpM = 64;
constexpr int kWarpN = 64;
constexpr int kThreadM = 8;
constexpr int kThreadN = 4;
constexpr int kWarpMIterations = 2;
constexpr int kWarpNIterations = 2;
constexpr int kWarpSubM = kWarpM / kWarpMIterations;
constexpr int kWarpSubN = kWarpN / kWarpNIterations;
constexpr int kWarps = (kBlockM / kWarpM) * (kBlockN / kWarpN);
constexpr int kThreads = kWarps * 32;
constexpr int kResultM = kWarpMIterations * kThreadM;
constexpr int kResultN = kWarpNIterations * kThreadN;

__global__ void warptiling_kernel(Problem problem,
                                  const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c) {
  __shared__ float tile_a[kBlockK * kBlockM];
  __shared__ float tile_b[kBlockK * kBlockN];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / 32;
  const int lane = thread % 32;
  const int warp_row = warp / (kBlockN / kWarpN);
  const int warp_column = warp % (kBlockN / kWarpN);
  const int thread_columns = kWarpSubN / kThreadN;
  const int thread_row = lane / thread_columns;
  const int thread_column = lane % thread_columns;
  float results[kResultM * kResultN] = {0.0F};

  for (int tile_start = 0; tile_start < problem.k; tile_start += kBlockK) {
    for (int vector_index = thread; vector_index < kBlockM * kBlockK / 4;
         vector_index += kThreads) {
      const int float_index = vector_index * 4;
      const int local_row = float_index / kBlockK;
      const int local_column = float_index % kBlockK;
      const int row = static_cast<int>(blockIdx.y) * kBlockM + local_row;
      const int column = tile_start + local_column;
      const float4 loaded = *reinterpret_cast<const float4*>(
          a + static_cast<std::size_t>(row) * problem.k + column);
      tile_a[(local_column + 0) * kBlockM + local_row] = loaded.x;
      tile_a[(local_column + 1) * kBlockM + local_row] = loaded.y;
      tile_a[(local_column + 2) * kBlockM + local_row] = loaded.z;
      tile_a[(local_column + 3) * kBlockM + local_row] = loaded.w;
    }
    for (int vector_index = thread; vector_index < kBlockK * kBlockN / 4;
         vector_index += kThreads) {
      const int float_index = vector_index * 4;
      const int local_row = float_index / kBlockN;
      const int local_column = float_index % kBlockN;
      const int row = tile_start + local_row;
      const int column = static_cast<int>(blockIdx.x) * kBlockN + local_column;
      *reinterpret_cast<float4*>(tile_b + float_index) = *reinterpret_cast<const float4*>(
          b + static_cast<std::size_t>(row) * problem.n + column);
    }
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kBlockK; ++inner) {
      float register_m[kResultM];
      float register_n[kResultN];
#pragma unroll
      for (int warp_m = 0; warp_m < kWarpMIterations; ++warp_m) {
#pragma unroll
        for (int row = 0; row < kThreadM; ++row) {
          register_m[warp_m * kThreadM + row] =
              tile_a[inner * kBlockM + warp_row * kWarpM + warp_m * kWarpSubM +
                     thread_row * kThreadM + row];
        }
      }
#pragma unroll
      for (int warp_n = 0; warp_n < kWarpNIterations; ++warp_n) {
#pragma unroll
        for (int column = 0; column < kThreadN; ++column) {
          register_n[warp_n * kThreadN + column] =
              tile_b[inner * kBlockN + warp_column * kWarpN + warp_n * kWarpSubN +
                     thread_column * kThreadN + column];
        }
      }
#pragma unroll
      for (int row = 0; row < kResultM; ++row) {
#pragma unroll
        for (int column = 0; column < kResultN; ++column) {
          results[row * kResultN + column] += register_m[row] * register_n[column];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int warp_m = 0; warp_m < kWarpMIterations; ++warp_m) {
#pragma unroll
    for (int row_offset = 0; row_offset < kThreadM; ++row_offset) {
      const int row = static_cast<int>(blockIdx.y) * kBlockM + warp_row * kWarpM +
                      warp_m * kWarpSubM + thread_row * kThreadM + row_offset;
#pragma unroll
      for (int warp_n = 0; warp_n < kWarpNIterations; ++warp_n) {
        const int column = static_cast<int>(blockIdx.x) * kBlockN + warp_column * kWarpN +
                           warp_n * kWarpSubN + thread_column * kThreadN;
        const auto index = static_cast<std::size_t>(row) * problem.n + column;
        const float4 old = *reinterpret_cast<const float4*>(c + index);
        const int result_index = (warp_m * kThreadM + row_offset) * kResultN +
                                 warp_n * kThreadN;
        const float4 output = make_float4(
            problem.alpha * results[result_index + 0] + problem.beta * old.x,
            problem.alpha * results[result_index + 1] + problem.beta * old.y,
            problem.alpha * results[result_index + 2] + problem.beta * old.z,
            problem.alpha * results[result_index + 3] + problem.beta * old.w);
        *reinterpret_cast<float4*>(c + index) = output;
      }
    }
  }
}

bool aligned16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

}  // namespace

LaunchResult launch_warptiling(const Problem& problem, const DeviceOperands& operands) {
  const bool compatible = problem.m % kBlockM == 0 && problem.n % kBlockN == 0 &&
                          problem.k % kBlockK == 0 && aligned16(operands.a) &&
                          aligned16(operands.b) && aligned16(operands.c);
  if (!compatible) {
    LaunchResult fallback = launch_blocktiling_2d(problem, operands);
    fallback.detail = "scalar-fallback";
    return fallback;
  }
  const dim3 block(kThreads);
  const dim3 grid(static_cast<unsigned int>(problem.n / kBlockN),
                  static_cast<unsigned int>(problem.m / kBlockM));
  warptiling_kernel<<<grid, block, 0, operands.stream>>>(
      problem, operands.a, operands.b, operands.c);
  return cuda_launch_result(cudaGetLastError(), "warp-tiled");
}

}  // namespace sgemm
