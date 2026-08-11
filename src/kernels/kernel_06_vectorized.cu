#include "sgemm/kernels.cuh"

#include <cstdint>

namespace sgemm {
namespace {

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 8;
constexpr int kThreadM = 8;
constexpr int kThreadN = 8;

__global__ void vectorized_kernel(Problem problem,
                                  const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c) {
  __shared__ float tile_a[kBlockK * kBlockM];
  __shared__ float tile_b[kBlockK * kBlockN];

  const int thread = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int c_row_start = static_cast<int>(blockIdx.y) * kBlockM +
                          static_cast<int>(threadIdx.y) * kThreadM;
  const int c_column_start = static_cast<int>(blockIdx.x) * kBlockN +
                             static_cast<int>(threadIdx.x) * kThreadN;
  float results[kThreadM * kThreadN] = {0.0F};

  for (int tile_start = 0; tile_start < problem.k; tile_start += kBlockK) {
    const int a_float = thread * 4;
    const int a_local_row = a_float / kBlockK;
    const int a_local_column = a_float % kBlockK;
    const int a_row = static_cast<int>(blockIdx.y) * kBlockM + a_local_row;
    const int a_column = tile_start + a_local_column;
    float4 loaded_a = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (a_row < problem.m && a_column + 3 < problem.k) {
      loaded_a = *reinterpret_cast<const float4*>(
          a + static_cast<std::size_t>(a_row) * problem.k + a_column);
    }
    tile_a[(a_local_column + 0) * kBlockM + a_local_row] = loaded_a.x;
    tile_a[(a_local_column + 1) * kBlockM + a_local_row] = loaded_a.y;
    tile_a[(a_local_column + 2) * kBlockM + a_local_row] = loaded_a.z;
    tile_a[(a_local_column + 3) * kBlockM + a_local_row] = loaded_a.w;

    const int b_float = thread * 4;
    const int b_local_row = b_float / kBlockN;
    const int b_local_column = b_float % kBlockN;
    const int b_row = tile_start + b_local_row;
    const int b_column = static_cast<int>(blockIdx.x) * kBlockN + b_local_column;
    float4 loaded_b = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (b_row < problem.k && b_column + 3 < problem.n) {
      loaded_b = *reinterpret_cast<const float4*>(
          b + static_cast<std::size_t>(b_row) * problem.n + b_column);
    }
    *reinterpret_cast<float4*>(tile_b + b_local_row * kBlockN + b_local_column) = loaded_b;
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kBlockK; ++inner) {
      float register_m[kThreadM];
      float register_n[kThreadN];
#pragma unroll
      for (int row = 0; row < kThreadM; ++row) {
        register_m[row] = tile_a[inner * kBlockM + threadIdx.y * kThreadM + row];
      }
#pragma unroll
      for (int column = 0; column < kThreadN; ++column) {
        register_n[column] = tile_b[inner * kBlockN + threadIdx.x * kThreadN + column];
      }
#pragma unroll
      for (int row = 0; row < kThreadM; ++row) {
#pragma unroll
        for (int column = 0; column < kThreadN; ++column) {
          results[row * kThreadN + column] += register_m[row] * register_n[column];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int row_offset = 0; row_offset < kThreadM; ++row_offset) {
    const int row = c_row_start + row_offset;
#pragma unroll
    for (int column_offset = 0; column_offset < kThreadN; column_offset += 4) {
      const int column = c_column_start + column_offset;
      if (row < problem.m && column + 3 < problem.n) {
        const auto index = static_cast<std::size_t>(row) * problem.n + column;
        const float4 old = *reinterpret_cast<const float4*>(c + index);
        const int result_index = row_offset * kThreadN + column_offset;
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

constexpr unsigned int ceil_div(unsigned int value, unsigned int divisor) {
  return (value + divisor - 1U) / divisor;
}

bool aligned16(const void* pointer) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignof(float4) == 0;
}

}  // namespace

LaunchResult launch_vectorized(const Problem& problem, const DeviceOperands& operands) {
  const bool vectorizable = problem.k % 4 == 0 && problem.n % 4 == 0 &&
                            aligned16(operands.a) && aligned16(operands.b) &&
                            aligned16(operands.c);
  if (!vectorizable) {
    LaunchResult fallback = launch_blocktiling_2d(problem, operands);
    fallback.detail = "scalar-fallback";
    return fallback;
  }
  const dim3 block(kBlockN / kThreadN, kBlockM / kThreadM);
  const dim3 grid(ceil_div(static_cast<unsigned int>(problem.n), kBlockN),
                  ceil_div(static_cast<unsigned int>(problem.m), kBlockM));
  vectorized_kernel<<<grid, block, 0, operands.stream>>>(
      problem, operands.a, operands.b, operands.c);
  return cuda_launch_result(cudaGetLastError(), "float4");
}

}  // namespace sgemm
