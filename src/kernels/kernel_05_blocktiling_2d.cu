#include "sgemm/kernels.cuh"
#include "sgemm/tutorial_config.hpp"

namespace sgemm {
namespace {

constexpr int kSafeBlockM = 64;
constexpr int kSafeBlockN = 64;
constexpr int kSafeBlockK = 8;
constexpr int kSafeThreadM = 8;
constexpr int kSafeThreadN = 8;
constexpr int kSafeThreads =
    (kSafeBlockM / kSafeThreadM) * (kSafeBlockN / kSafeThreadN);

__global__ void blocktiling_2d_tail_safe_kernel(Problem problem,
                                                const float* __restrict__ a,
                                                const float* __restrict__ b,
                                                float* __restrict__ c) {
  __shared__ float tile_a[kSafeBlockM * kSafeBlockK];
  __shared__ float tile_b[kSafeBlockK * kSafeBlockN];

  const int thread = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int c_row_start = static_cast<int>(blockIdx.y) * kSafeBlockM +
                          static_cast<int>(threadIdx.y) * kSafeThreadM;
  const int c_column_start = static_cast<int>(blockIdx.x) * kSafeBlockN +
                             static_cast<int>(threadIdx.x) * kSafeThreadN;
  float results[kSafeThreadM * kSafeThreadN] = {0.0F};

  for (int tile_start = 0; tile_start < problem.k; tile_start += kSafeBlockK) {
    for (int index = thread; index < kSafeBlockM * kSafeBlockK;
         index += kSafeThreads) {
      const int local_row = index / kSafeBlockK;
      const int local_column = index % kSafeBlockK;
      const int row = static_cast<int>(blockIdx.y) * kSafeBlockM + local_row;
      const int column = tile_start + local_column;
      tile_a[index] = row < problem.m && column < problem.k
                          ? a[static_cast<std::size_t>(row) * problem.k + column]
                          : 0.0F;
    }
    for (int index = thread; index < kSafeBlockK * kSafeBlockN;
         index += kSafeThreads) {
      const int local_row = index / kSafeBlockN;
      const int local_column = index % kSafeBlockN;
      const int row = tile_start + local_row;
      const int column = static_cast<int>(blockIdx.x) * kSafeBlockN + local_column;
      tile_b[index] = row < problem.k && column < problem.n
                          ? b[static_cast<std::size_t>(row) * problem.n + column]
                          : 0.0F;
    }
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kSafeBlockK; ++inner) {
      float register_m[kSafeThreadM];
      float register_n[kSafeThreadN];
#pragma unroll
      for (int row = 0; row < kSafeThreadM; ++row) {
        register_m[row] =
            tile_a[(threadIdx.y * kSafeThreadM + row) * kSafeBlockK + inner];
      }
#pragma unroll
      for (int column = 0; column < kSafeThreadN; ++column) {
        register_n[column] =
            tile_b[inner * kSafeBlockN + threadIdx.x * kSafeThreadN + column];
      }
#pragma unroll
      for (int row = 0; row < kSafeThreadM; ++row) {
#pragma unroll
        for (int column = 0; column < kSafeThreadN; ++column) {
          results[row * kSafeThreadN + column] +=
              register_m[row] * register_n[column];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int row_offset = 0; row_offset < kSafeThreadM; ++row_offset) {
#pragma unroll
    for (int column_offset = 0; column_offset < kSafeThreadN; ++column_offset) {
      const int row = c_row_start + row_offset;
      const int column = c_column_start + column_offset;
      if (row < problem.m && column < problem.n) {
        const auto index = static_cast<std::size_t>(row) * problem.n + column;
        c[index] =
            problem.alpha * results[row_offset * kSafeThreadN + column_offset] +
            problem.beta * c[index];
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    blocktiling_2d_fast_kernel(Problem problem,
                               const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c) {
  constexpr int threads = (BM * BN) / (TM * TN);
  static_assert(BM % TM == 0 && BN % TN == 0);

  __shared__ float tile_a[BM * BK];
  __shared__ float tile_b[BK * BN];

  const int thread = static_cast<int>(threadIdx.x);
  const int thread_column = thread % (BN / TN);
  const int thread_row = thread / (BN / TN);
  const int block_row = static_cast<int>(blockIdx.y) * BM;
  const int block_column = static_cast<int>(blockIdx.x) * BN;
  const int inner_row_a = thread / BK;
  const int inner_column_a = thread % BK;
  const int stride_a = threads / BK;
  const int inner_row_b = thread / BN;
  const int inner_column_b = thread % BN;
  const int stride_b = threads / BN;

  float results[TM * TN] = {0.0F};
  float register_m[TM] = {0.0F};
  float register_n[TN] = {0.0F};

  for (int tile_start = 0; tile_start < problem.k; tile_start += BK) {
#pragma unroll
    for (int load_offset = 0; load_offset < BM; load_offset += stride_a) {
      const int local_row = inner_row_a + load_offset;
      tile_a[local_row * BK + inner_column_a] =
          a[static_cast<std::size_t>(block_row + local_row) * problem.k + tile_start +
            inner_column_a];
    }
#pragma unroll
    for (int load_offset = 0; load_offset < BK; load_offset += stride_b) {
      const int local_row = inner_row_b + load_offset;
      tile_b[local_row * BN + inner_column_b] =
          b[static_cast<std::size_t>(tile_start + local_row) * problem.n + block_column +
            inner_column_b];
    }
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < BK; ++inner) {
#pragma unroll
      for (int row = 0; row < TM; ++row) {
        register_m[row] = tile_a[(thread_row * TM + row) * BK + inner];
      }
#pragma unroll
      for (int column = 0; column < TN; ++column) {
        register_n[column] =
            tile_b[inner * BN + thread_column * TN + column];
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
    for (int column_offset = 0; column_offset < TN; ++column_offset) {
      const int row = block_row + thread_row * TM + row_offset;
      const int column = block_column + thread_column * TN + column_offset;
      const auto index = static_cast<std::size_t>(row) * problem.n + column;
      c[index] = problem.alpha * results[row_offset * TN + column_offset] +
                 problem.beta * c[index];
    }
  }
}

constexpr unsigned int ceil_div(unsigned int value, unsigned int divisor) {
  return (value + divisor - 1U) / divisor;
}

}  // namespace

LaunchResult launch_blocktiling_2d(const Problem& problem, const DeviceOperands& operands) {
  constexpr auto config = tutorial::kKernel5Fast;
  const bool compatible = problem.m % config.block_m == 0 &&
                          problem.n % config.block_n == 0 &&
                          problem.k % config.block_k == 0;
  if (compatible) {
    constexpr int threads =
        (config.block_m * config.block_n) / (config.thread_m * config.thread_n);
    const dim3 block(threads);
    const dim3 grid(static_cast<unsigned int>(problem.n / config.block_n),
                    static_cast<unsigned int>(problem.m / config.block_m));
    blocktiling_2d_fast_kernel<config.block_m,
                               config.block_n,
                               config.block_k,
                               config.thread_m,
                               config.thread_n><<<grid, block, 0, operands.stream>>>(
        problem, operands.a, operands.b, operands.c);
    return cuda_launch_result(cudaGetLastError(), "register-tile-2d-fast");
  }

  const dim3 block(kSafeBlockN / kSafeThreadN, kSafeBlockM / kSafeThreadM);
  const dim3 grid(ceil_div(static_cast<unsigned int>(problem.n), kSafeBlockN),
                  ceil_div(static_cast<unsigned int>(problem.m), kSafeBlockM));
  blocktiling_2d_tail_safe_kernel<<<grid, block, 0, operands.stream>>>(
      problem, operands.a, operands.b, operands.c);
  return cuda_launch_result(cudaGetLastError(), "tail-safe");
}

}  // namespace sgemm
