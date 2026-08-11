#include "sgemm/kernels.cuh"

namespace sgemm {
namespace {

__global__ void coalesced_kernel(Problem problem,
                                 const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c) {
  constexpr int tile = 32;
  const int thread = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.y) * tile + thread / tile;
  const int column = static_cast<int>(blockIdx.x) * tile + thread % tile;
  if (row >= problem.m || column >= problem.n) {
    return;
  }

  float sum = 0.0F;
  for (int inner = 0; inner < problem.k; ++inner) {
    sum += a[static_cast<std::size_t>(row) * problem.k + inner] *
           b[static_cast<std::size_t>(inner) * problem.n + column];
  }
  const auto index = static_cast<std::size_t>(row) * problem.n + column;
  c[index] = problem.alpha * sum + problem.beta * c[index];
}

constexpr unsigned int ceil_div(unsigned int value, unsigned int divisor) {
  return (value + divisor - 1U) / divisor;
}

}  // namespace

LaunchResult launch_coalesced(const Problem& problem, const DeviceOperands& operands) {
  constexpr unsigned int tile = 32;
  const dim3 block(tile * tile);
  const dim3 grid(ceil_div(static_cast<unsigned int>(problem.n), tile),
                  ceil_div(static_cast<unsigned int>(problem.m), tile));
  coalesced_kernel<<<grid, block, 0, operands.stream>>>(
      problem, operands.a, operands.b, operands.c);
  return cuda_launch_result(cudaGetLastError(), "coalesced-scalar");
}

}  // namespace sgemm
