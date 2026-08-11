#include "sgemm/kernels.cuh"

#include <string>

namespace sgemm {

std::vector<KernelSpec> phase_a_kernels() {
  return {
      {1, "naive", "One output element per thread", launch_naive},
      {2, "coalesced", "Coalesced global-memory access", launch_coalesced},
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
