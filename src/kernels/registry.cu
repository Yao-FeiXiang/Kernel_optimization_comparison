#include "sgemm/kernels.cuh"

#include <string>

namespace sgemm {

std::vector<KernelSpec> phase_a_kernels() {
  return {
      {1, "naive", "One output element per thread", launch_naive},
      {2, "coalesced", "Coalesced global-memory access", launch_coalesced},
      {3, "shared", "Shared-memory cache blocking", launch_shared},
      {4, "blocktiling-1d", "Eight output rows per thread", launch_blocktiling_1d},
      {5, "blocktiling-2d", "An 8x8 output tile per thread", launch_blocktiling_2d},
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
