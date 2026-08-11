#include "sgemm/benchmark.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void test_cpu_sgemm_uses_alpha_and_beta() {
  const sgemm::Problem problem{2, 2, 3, 0.5F, 0.25F};
  const std::vector<float> a{1, 2, 3, 4, 5, 6};
  const std::vector<float> b{7, 8, 9, 10, 11, 12};
  const std::vector<float> c0{4, 8, 12, 16};
  const auto got = sgemm::cpu_sgemm(problem, a, b, c0);
  const std::vector<float> want{30, 34, 72.5F, 81};
  assert(sgemm::compare_vectors(got, want, 1e-6F, 1e-6F).passed);
}

void test_inputs_are_reproducible_and_seeded() {
  const sgemm::Problem problem{3, 4, 2, 0.8F, 0.2F};
  const auto first = sgemm::make_inputs(problem, 7);
  const auto second = sgemm::make_inputs(problem, 7);
  const auto other = sgemm::make_inputs(problem, 8);
  assert(first.a == second.a);
  assert(first.b == second.b);
  assert(first.c0 == second.c0);
  assert(first.a != other.a);
}

void test_compare_vectors_uses_combined_tolerance() {
  const auto accepted = sgemm::compare_vectors({1000.0F}, {1000.05F}, 0.001F, 0.0001F);
  assert(accepted.passed);
  const auto rejected = sgemm::compare_vectors({1.0F}, {1.1F}, 0.001F, 0.001F);
  assert(!rejected.passed);
  assert(rejected.worst_index == 0);
  assert(rejected.max_abs_error > 0.09F);
}

void test_operation_count_matches_documented_formula() {
  const sgemm::Problem problem{2, 2, 3, 0.5F, 0.25F};
  assert(sgemm::operation_count(problem) == 32.0);
}

void test_invalid_input_sizes_are_rejected() {
  const sgemm::Problem problem{2, 2, 3, 1.0F, 0.0F};
  bool raised = false;
  try {
    static_cast<void>(sgemm::cpu_sgemm(problem, {1.0F}, {1.0F}, {1.0F}));
  } catch (const std::invalid_argument&) {
    raised = true;
  }
  assert(raised);
}

}  // namespace

int main() {
  test_cpu_sgemm_uses_alpha_and_beta();
  test_inputs_are_reproducible_and_seeded();
  test_compare_vectors_uses_combined_tolerance();
  test_operation_count_matches_documented_formula();
  test_invalid_input_sizes_are_rejected();
  return 0;
}
