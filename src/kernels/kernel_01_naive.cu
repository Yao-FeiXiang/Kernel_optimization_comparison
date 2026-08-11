#include "sgemm/kernels.cuh"

namespace sgemm {
namespace {

__global__ void naive_kernel(Problem problem,
                             const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c) {
  const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
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

LaunchResult launch_naive(const Problem& problem, const DeviceOperands& operands) {
  const dim3 block(32, 32);
  const dim3 grid(ceil_div(static_cast<unsigned int>(problem.n), block.x),
                  ceil_div(static_cast<unsigned int>(problem.m), block.y));
  naive_kernel<<<grid, block, 0, operands.stream>>>(problem, operands.a, operands.b, operands.c);
  return cuda_launch_result(cudaGetLastError(), "scalar");
}

}  // namespace sgemm
