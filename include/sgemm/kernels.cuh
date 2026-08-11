#pragma once

#include "sgemm/benchmark.hpp"

#include <cuda_runtime.h>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sgemm {

struct DeviceOperands {
  const float* a;
  const float* b;
  const float* c0;
  float* c;
  cudaStream_t stream;
};

using PrepareFn = PrepareResult (*)(const Problem&, const DeviceOperands&);
using LaunchFn = LaunchResult (*)(const Problem&, const DeviceOperands&);
using CleanupFn = void (*)();

struct KernelSpec {
  int id;
  std::string_view name;
  std::string_view description;
  PrepareFn prepare;
  LaunchFn launch;
  CleanupFn cleanup;
};

inline LaunchResult cuda_launch_result(cudaError_t status, std::string detail = {}) {
  if (status == cudaSuccess) {
    return {true, {}, std::move(detail)};
  }
  return {false, cudaGetErrorString(status), std::move(detail)};
}

LaunchResult launch_naive(const Problem& problem, const DeviceOperands& operands);

LaunchResult launch_coalesced(const Problem& problem, const DeviceOperands& operands);

LaunchResult launch_shared(const Problem& problem, const DeviceOperands& operands);

LaunchResult launch_blocktiling_1d(const Problem& problem, const DeviceOperands& operands);

LaunchResult launch_blocktiling_2d(const Problem& problem, const DeviceOperands& operands);

LaunchResult launch_vectorized(const Problem& problem, const DeviceOperands& operands);

PrepareResult prepare_autotuned(const Problem& problem, const DeviceOperands& operands);
LaunchResult launch_autotuned(const Problem& problem, const DeviceOperands& operands);

std::vector<KernelSpec> phase_a_kernels();
std::optional<KernelSpec> find_kernel(std::string_view selector);

}  // namespace sgemm
