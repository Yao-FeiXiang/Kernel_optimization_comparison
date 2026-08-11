#include "sgemm/kernels.cuh"

#include <string>

namespace sgemm {
namespace {

PrepareResult fixed_prepare(const Problem&, const DeviceOperands&) {
  return {true, "fixed", {}};
}

void fixed_cleanup() {}

}  // namespace

std::vector<KernelSpec> phase_a_kernels() {
  return {
      {1, "naive", "One output element per thread", fixed_prepare, launch_naive, fixed_cleanup},
      {2, "coalesced", "Coalesced global-memory access", fixed_prepare, launch_coalesced,
       fixed_cleanup},
      {3, "shared", "Shared-memory cache blocking", fixed_prepare, launch_shared, fixed_cleanup},
      {4, "blocktiling-1d", "Eight output rows per thread", fixed_prepare,
       launch_blocktiling_1d, fixed_cleanup},
      {5, "blocktiling-2d", "An 8x8 output tile per thread", fixed_prepare,
       launch_blocktiling_2d, fixed_cleanup},
  };
}

std::optional<KernelSpec> find_kernel(std::string_view selector) {
  for (const auto& kernel : phase_a_kernels()) {
    if (selector == kernel.name || selector == std::to_string(kernel.id)) {
      return kernel;
    }
  }
  return std::nullopt;
}

}  // namespace sgemm
