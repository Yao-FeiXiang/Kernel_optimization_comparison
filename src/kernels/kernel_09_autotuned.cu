#include "sgemm/kernels.cuh"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace sgemm {
namespace {

template <int BM, int BN, int BK, int TM, int TN>
__global__ void tuned_kernel(Problem problem,
                             const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c) {
  static_assert(BM % TM == 0 && BN % TN == 0);
  constexpr int threads = (BM / TM) * (BN / TN);
  static_assert(threads <= 1024);
  __shared__ float tile_a[BK * BM];
  __shared__ float tile_b[BK * BN];

  const int thread = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int c_row_start = static_cast<int>(blockIdx.y) * BM +
                          static_cast<int>(threadIdx.y) * TM;
  const int c_column_start = static_cast<int>(blockIdx.x) * BN +
                             static_cast<int>(threadIdx.x) * TN;
  float results[TM * TN] = {0.0F};

  for (int tile_start = 0; tile_start < problem.k; tile_start += BK) {
    for (int index = thread; index < BM * BK; index += threads) {
      const int local_row = index / BK;
      const int local_column = index % BK;
      const int row = static_cast<int>(blockIdx.y) * BM + local_row;
      const int column = tile_start + local_column;
      tile_a[local_column * BM + local_row] =
          row < problem.m && column < problem.k
              ? a[static_cast<std::size_t>(row) * problem.k + column]
              : 0.0F;
    }
    for (int index = thread; index < BK * BN; index += threads) {
      const int local_row = index / BN;
      const int local_column = index % BN;
      const int row = tile_start + local_row;
      const int column = static_cast<int>(blockIdx.x) * BN + local_column;
      tile_b[index] = row < problem.k && column < problem.n
                          ? b[static_cast<std::size_t>(row) * problem.n + column]
                          : 0.0F;
    }
    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < BK; ++inner) {
      float register_m[TM];
      float register_n[TN];
#pragma unroll
      for (int row = 0; row < TM; ++row) {
        register_m[row] = tile_a[inner * BM + threadIdx.y * TM + row];
      }
#pragma unroll
      for (int column = 0; column < TN; ++column) {
        register_n[column] = tile_b[inner * BN + threadIdx.x * TN + column];
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
      const int row = c_row_start + row_offset;
      const int column = c_column_start + column_offset;
      if (row < problem.m && column < problem.n) {
        const auto index = static_cast<std::size_t>(row) * problem.n + column;
        c[index] = problem.alpha * results[row_offset * TN + column_offset] +
                   problem.beta * c[index];
      }
    }
  }
}

constexpr unsigned int ceil_div(unsigned int value, unsigned int divisor) {
  return (value + divisor - 1U) / divisor;
}

template <int BM, int BN, int BK, int TM, int TN>
LaunchResult launch_candidate(const Problem& problem, const DeviceOperands& operands) {
  const dim3 block(BN / TN, BM / TM);
  const dim3 grid(ceil_div(static_cast<unsigned int>(problem.n), BN),
                  ceil_div(static_cast<unsigned int>(problem.m), BM));
  tuned_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, operands.stream>>>(
      problem, operands.a, operands.b, operands.c);
  return cuda_launch_result(cudaGetLastError(), "autotuned-tile");
}

struct Candidate {
  const char* configuration;
  LaunchFn launch;
};

const std::array<Candidate, 5> kCandidates{{
    {"BM64_BN64_BK8_TM8_TN8", launch_candidate<64, 64, 8, 8, 8>},
    {"BM128_BN64_BK8_TM8_TN8", launch_candidate<128, 64, 8, 8, 8>},
    {"BM64_BN128_BK8_TM8_TN8", launch_candidate<64, 128, 8, 8, 8>},
    {"BM128_BN128_BK8_TM8_TN8", launch_candidate<128, 128, 8, 8, 8>},
    {"BM128_BN128_BK16_TM8_TN8", launch_candidate<128, 128, 16, 8, 8>},
}};

struct TuneCache {
  int device{-1};
  int m{-1};
  int n{-1};
  int k{-1};
  int selected{-1};
};

TuneCache g_cache;

PrepareResult cuda_prepare_error(cudaError_t status, const char* operation) {
  return {false, true, {}, std::string(operation) + ": " + cudaGetErrorString(status)};
}

}  // namespace

PrepareResult prepare_autotuned(const Problem& problem, const DeviceOperands& operands) {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return cuda_prepare_error(status, "cudaGetDevice");
  }
  if (g_cache.device == device && g_cache.m == problem.m && g_cache.n == problem.n &&
      g_cache.k == problem.k && g_cache.selected >= 0) {
    return {true, true, kCandidates[static_cast<std::size_t>(g_cache.selected)].configuration,
            {}};
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
  for (std::size_t candidate_index = 0; candidate_index < kCandidates.size(); ++candidate_index) {
    std::array<float, 3> samples{};
    status = cudaMemcpyAsync(
        operands.c, operands.c0, c_bytes, cudaMemcpyDeviceToDevice, operands.stream);
    LaunchResult warmup = kCandidates[candidate_index].launch(problem, operands);
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
      const LaunchResult launched = kCandidates[candidate_index].launch(problem, operands);
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
  return {true, true, kCandidates[static_cast<std::size_t>(best_index)].configuration, {}};
}

LaunchResult launch_autotuned(const Problem& problem, const DeviceOperands& operands) {
  if (g_cache.selected < 0 || g_cache.selected >= static_cast<int>(kCandidates.size())) {
    return {false, "autotuned method was launched before preparation", {}};
  }
  return kCandidates[static_cast<std::size_t>(g_cache.selected)].launch(problem, operands);
}

}  // namespace sgemm
