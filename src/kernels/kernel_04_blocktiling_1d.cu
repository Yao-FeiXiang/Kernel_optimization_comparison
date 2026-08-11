#include "sgemm/kernels.cuh"

namespace sgemm {
namespace {

constexpr int kBlockM = 64;
constexpr int kBlockN = 64;
constexpr int kBlockK = 8;
constexpr int kThreadM = 8;

__global__ void blocktiling_1d_kernel(Problem problem,
                                      const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ c) {
  __shared__ float tile_a[kBlockM * kBlockK];
  __shared__ float tile_b[kBlockK * kBlockN];

  const int thread = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int a_load_row = thread / kBlockK;
  const int a_load_column = thread % kBlockK;
  const int b_load_row = thread / kBlockN;
  const int b_load_column = thread % kBlockN;
  const int c_row_start = static_cast<int>(blockIdx.y) * kBlockM +
                          static_cast<int>(threadIdx.y) * kThreadM;
  const int c_column = static_cast<int>(blockIdx.x) * kBlockN +
                       static_cast<int>(threadIdx.x);
  float results[kThreadM] = {0.0F};

  for (int tile_start = 0; tile_start < problem.k; tile_start += kBlockK) {
    const int a_row = static_cast<int>(blockIdx.y) * kBlockM + a_load_row;
    const int a_column = tile_start + a_load_column;
    const int b_row = tile_start + b_load_row;
    const int b_column = static_cast<int>(blockIdx.x) * kBlockN + b_load_column;
    tile_a[a_load_row * kBlockK + a_load_column] =
        a_row < problem.m && a_column < problem.k
            ? a[static_cast<std::size_t>(a_row) * problem.k + a_column]
            : 0.0F;
    tile_b[b_load_row * kBlockN + b_load_column] =
        b_row < problem.k && b_column < problem.n
            ? b[static_cast<std::size_t>(b_row) * problem.n + b_column]
            : 0.0F;
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kBlockK; ++inner) {
      const float b_value = tile_b[inner * kBlockN + threadIdx.x];
#pragma unroll
      for (int result = 0; result < kThreadM; ++result) {
        results[result] +=
            tile_a[(threadIdx.y * kThreadM + result) * kBlockK + inner] * b_value;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int result = 0; result < kThreadM; ++result) {
    const int row = c_row_start + result;
    if (row < problem.m && c_column < problem.n) {
      const auto index = static_cast<std::size_t>(row) * problem.n + c_column;
      c[index] = problem.alpha * results[result] + problem.beta * c[index];
    }
  }
}

constexpr unsigned int ceil_div(unsigned int value, unsigned int divisor) {
  return (value + divisor - 1U) / divisor;
}

}  // namespace

cudaError_t launch_blocktiling_1d(const Problem& problem,
                                  const float* a,
                                  const float* b,
                                  float* c,
                                  cudaStream_t stream) {
  const dim3 block(kBlockN, kBlockM / kThreadM);
  const dim3 grid(ceil_div(static_cast<unsigned int>(problem.n), kBlockN),
                  ceil_div(static_cast<unsigned int>(problem.m), kBlockM));
  blocktiling_1d_kernel<<<grid, block, 0, stream>>>(problem, a, b, c);
  return cudaGetLastError();
}

}  // namespace sgemm
