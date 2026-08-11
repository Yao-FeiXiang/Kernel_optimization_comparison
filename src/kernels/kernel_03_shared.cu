#include "sgemm/kernels.cuh"

namespace sgemm {
namespace {

constexpr int kBlock = 32;

__global__ void shared_kernel(Problem problem,
                              const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c) {
  __shared__ float tile_a[kBlock * kBlock];
  __shared__ float tile_b[kBlock * kBlock];

  const int local_row = static_cast<int>(threadIdx.y);
  const int local_column = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.y) * kBlock + local_row;
  const int column = static_cast<int>(blockIdx.x) * kBlock + local_column;
  float sum = 0.0F;

  for (int tile_start = 0; tile_start < problem.k; tile_start += kBlock) {
    const int a_column = tile_start + local_column;
    const int b_row = tile_start + local_row;
    tile_a[local_row * kBlock + local_column] =
        row < problem.m && a_column < problem.k
            ? a[static_cast<std::size_t>(row) * problem.k + a_column]
            : 0.0F;
    tile_b[local_row * kBlock + local_column] =
        b_row < problem.k && column < problem.n
            ? b[static_cast<std::size_t>(b_row) * problem.n + column]
            : 0.0F;
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kBlock; ++inner) {
      sum += tile_a[local_row * kBlock + inner] *
             tile_b[inner * kBlock + local_column];
    }
    __syncthreads();
  }

  if (row < problem.m && column < problem.n) {
    const auto index = static_cast<std::size_t>(row) * problem.n + column;
    c[index] = problem.alpha * sum + problem.beta * c[index];
  }
}

constexpr unsigned int ceil_div(unsigned int value, unsigned int divisor) {
  return (value + divisor - 1U) / divisor;
}

}  // namespace

LaunchResult launch_shared(const Problem& problem, const DeviceOperands& operands) {
  const dim3 block(kBlock, kBlock);
  const dim3 grid(ceil_div(static_cast<unsigned int>(problem.n), kBlock),
                  ceil_div(static_cast<unsigned int>(problem.m), kBlock));
  shared_kernel<<<grid, block, 0, operands.stream>>>(
      problem, operands.a, operands.b, operands.c);
  return cuda_launch_result(cudaGetLastError(), "shared-memory");
}

}  // namespace sgemm
