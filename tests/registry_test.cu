#include "sgemm/kernels.cuh"

#include <cassert>
#include <set>
#include <string>

int main() {
  const auto kernels = sgemm::phase_a_kernels();
  assert(kernels.size() == 7);
  assert(sgemm::find_kernel("1").has_value());
  assert(sgemm::find_kernel("1")->name == "naive");
  assert(sgemm::find_kernel("naive")->id == 1);
  assert(sgemm::find_kernel("2")->name == "coalesced");
  assert(sgemm::find_kernel("coalesced")->id == 2);
  assert(sgemm::find_kernel("3")->name == "shared");
  assert(sgemm::find_kernel("shared")->id == 3);
  assert(sgemm::find_kernel("4")->name == "blocktiling-1d");
  assert(sgemm::find_kernel("blocktiling-1d")->id == 4);
  assert(sgemm::find_kernel("5")->name == "blocktiling-2d");
  assert(sgemm::find_kernel("blocktiling-2d")->id == 5);
  assert(sgemm::find_kernel("6")->name == "vectorized");
  assert(sgemm::find_kernel("vectorized")->id == 6);
  assert(sgemm::find_kernel("9")->name == "autotuned");
  assert(sgemm::find_kernel("autotuned")->id == 9);
  assert(!sgemm::find_kernel("missing").has_value());

  std::set<int> ids;
  std::set<std::string> names;
  for (const auto& kernel : kernels) {
    assert(kernel.prepare != nullptr);
    assert(kernel.launch != nullptr);
    assert(kernel.cleanup != nullptr);
    assert(!kernel.description.empty());
    ids.insert(kernel.id);
    names.emplace(kernel.name);
  }
  assert(ids.size() == kernels.size());
  assert(names.size() == kernels.size());
  return 0;
}
