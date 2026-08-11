#pragma once

#include "sgemm/benchmark.hpp"

#include <cuda_runtime.h>

#include <optional>
#include <string_view>
#include <vector>

namespace sgemm {

using LaunchFn = cudaError_t (*)(const Problem&,
                                 const float*,
                                 const float*,
                                 float*,
                                 cudaStream_t);

struct KernelSpec {
  int id;
  std::string_view name;
  std::string_view description;
  LaunchFn launch;
};

cudaError_t launch_naive(const Problem& problem,
                         const float* a,
                         const float* b,
                         float* c,
                         cudaStream_t stream);

cudaError_t launch_coalesced(const Problem& problem,
                             const float* a,
                             const float* b,
                             float* c,
                             cudaStream_t stream);

cudaError_t launch_shared(const Problem& problem,
                          const float* a,
                          const float* b,
                          float* c,
                          cudaStream_t stream);

cudaError_t launch_blocktiling_1d(const Problem& problem,
                                  const float* a,
                                  const float* b,
                                  float* c,
                                  cudaStream_t stream);

std::vector<KernelSpec> phase_a_kernels();
std::optional<KernelSpec> find_kernel(std::string_view selector);

}  // namespace sgemm
