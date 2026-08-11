#include "sgemm/kernels.cuh"

#include <cassert>
#include <set>
#include <string>

int main() {
  const auto kernels = sgemm::phase_a_kernels();
  assert(kernels.size() == 2);
  assert(sgemm::find_kernel("1").has_value());
  assert(sgemm::find_kernel("1")->name == "naive");
  assert(sgemm::find_kernel("naive")->id == 1);
  assert(sgemm::find_kernel("2")->name == "coalesced");
  assert(sgemm::find_kernel("coalesced")->id == 2);
  assert(!sgemm::find_kernel("missing").has_value());

  std::set<int> ids;
  std::set<std::string> names;
  for (const auto& kernel : kernels) {
    assert(kernel.launch != nullptr);
    assert(!kernel.description.empty());
    ids.insert(kernel.id);
    names.emplace(kernel.name);
  }
  assert(ids.size() == kernels.size());
  assert(names.size() == kernels.size());
  return 0;
}
