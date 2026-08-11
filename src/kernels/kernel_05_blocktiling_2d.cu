#include "sgemm/kernels.cuh"

namespace sgemm {
namespace {

constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kBlockK = 8;
constexpr int kThreadM = 8;
constexpr int kThreadN = 8;
constexpr int kThreads = (kBlockM / kThreadM) * (kBlockN / kThreadN);

__global__ void blocktiling_2d_kernel(Problem problem,
                                      const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ c) {
  __shared__ float tile_a[kBlockM * kBlockK];
  __shared__ float tile_b[kBlockK * kBlockN];

  const int thread = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int c_row_start = static_cast<int>(blockIdx.y) * kBlockM +
                          static_cast<int>(threadIdx.y) * kThreadM;
  const int c_column_start = static_cast<int>(blockIdx.x) * kBlockN +
                             static_cast<int>(threadIdx.x) * kThreadN;
  float results[kThreadM * kThreadN] = {0.0F};

  for (int tile_start = 0; tile_start < problem.k; tile_start += kBlockK) {
    for (int index = thread; index < kBlockM * kBlockK; index += kThreads) {
      const int local_row = index / kBlockK;
      const int local_column = index % kBlockK;
      const int row = static_cast<int>(blockIdx.y) * kBlockM + local_row;
      const int column = tile_start + local_column;
      tile_a[index] = row < problem.m && column < problem.k
                          ? a[static_cast<std::size_t>(row) * problem.k + column]
                          : 0.0F;
    }
    for (int index = thread; index < kBlockK * kBlockN; index += kThreads) {
      const int local_row = index / kBlockN;
      const int local_column = index % kBlockN;
      const int row = tile_start + local_row;
      const int column = static_cast<int>(blockIdx.x) * kBlockN + local_column;
      tile_b[index] = row < problem.k && column < problem.n
                          ? b[static_cast<std::size_t>(row) * problem.n + column]
                          : 0.0F;
    }
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kBlockK; ++inner) {
      float register_m[kThreadM];
      float register_n[kThreadN];
#pragma unroll
      for (int row = 0; row < kThreadM; ++row) {
        register_m[row] = tile_a[(threadIdx.y * kThreadM + row) * kBlockK + inner];
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
#pragma unroll
    for (int column_offset = 0; column_offset < kThreadN; ++column_offset) {
      const int row = c_row_start + row_offset;
      const int column = c_column_start + column_offset;
      if (row < problem.m && column < problem.n) {
        const auto index = static_cast<std::size_t>(row) * problem.n + column;
        c[index] = problem.alpha * results[row_offset * kThreadN + column_offset] +
                   problem.beta * c[index];
      }
    }
  }
}

constexpr unsigned int ceil_div(unsigned int value, unsigned int divisor) {
  return (value + divisor - 1U) / divisor;
}

}  // namespace

LaunchResult launch_blocktiling_2d(const Problem& problem, const DeviceOperands& operands) {
  const dim3 block(kBlockN / kThreadN, kBlockM / kThreadM);
  const dim3 grid(ceil_div(static_cast<unsigned int>(problem.n), kBlockN),
                  ceil_div(static_cast<unsigned int>(problem.m), kBlockM));
  blocktiling_2d_kernel<<<grid, block, 0, operands.stream>>>(
      problem, operands.a, operands.b, operands.c);
  return cuda_launch_result(cudaGetLastError(), "register-tile-2d");
}

}  // namespace sgemm
