#include "sgemm/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace sgemm {
namespace {

std::size_t matrix_elements(int rows, int columns, const char* name) {
  if (rows <= 0 || columns <= 0) {
    throw std::invalid_argument(std::string(name) + " dimensions must be positive");
  }
  const auto unsigned_rows = static_cast<std::size_t>(rows);
  const auto unsigned_columns = static_cast<std::size_t>(columns);
  if (unsigned_rows > std::numeric_limits<std::size_t>::max() / unsigned_columns) {
    throw std::overflow_error(std::string(name) + " element count overflows size_t");
  }
  return unsigned_rows * unsigned_columns;
}

void validate_problem(const Problem& problem) {
  static_cast<void>(matrix_elements(problem.m, problem.k, "A"));
  static_cast<void>(matrix_elements(problem.k, problem.n, "B"));
  static_cast<void>(matrix_elements(problem.m, problem.n, "C"));
  if (!std::isfinite(problem.alpha) || !std::isfinite(problem.beta)) {
    throw std::invalid_argument("alpha and beta must be finite");
  }
}

}  // namespace

InputMatrices make_inputs(const Problem& problem, std::uint32_t seed) {
  validate_problem(problem);
  InputMatrices inputs;
  inputs.a.resize(matrix_elements(problem.m, problem.k, "A"));
  inputs.b.resize(matrix_elements(problem.k, problem.n, "B"));
  inputs.c0.resize(matrix_elements(problem.m, problem.n, "C"));

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  const auto fill = [&](std::vector<float>& values) {
    std::generate(values.begin(), values.end(), [&] { return distribution(generator); });
  };
  fill(inputs.a);
  fill(inputs.b);
  fill(inputs.c0);
  return inputs;
}

std::vector<float> cpu_sgemm(const Problem& problem,
                             const std::vector<float>& a,
                             const std::vector<float>& b,
                             const std::vector<float>& c0) {
  validate_problem(problem);
  const auto a_size = matrix_elements(problem.m, problem.k, "A");
  const auto b_size = matrix_elements(problem.k, problem.n, "B");
  const auto c_size = matrix_elements(problem.m, problem.n, "C");
  if (a.size() != a_size || b.size() != b_size || c0.size() != c_size) {
    throw std::invalid_argument("input matrix storage does not match the problem dimensions");
  }

  std::vector<float> result(c_size);
  for (int row = 0; row < problem.m; ++row) {
    for (int column = 0; column < problem.n; ++column) {
      float sum = 0.0F;
      for (int inner = 0; inner < problem.k; ++inner) {
        sum += a[static_cast<std::size_t>(row) * problem.k + inner] *
               b[static_cast<std::size_t>(inner) * problem.n + column];
      }
      const auto index = static_cast<std::size_t>(row) * problem.n + column;
      result[index] = problem.alpha * sum + problem.beta * c0[index];
    }
  }
  return result;
}

ErrorStats compare_vectors(const std::vector<float>& actual,
                           const std::vector<float>& reference,
                           float atol,
                           float rtol) {
  if (actual.size() != reference.size()) {
    throw std::invalid_argument("actual and reference vectors must have equal sizes");
  }
  if (atol < 0.0F || rtol < 0.0F || !std::isfinite(atol) || !std::isfinite(rtol)) {
    throw std::invalid_argument("comparison tolerances must be finite and nonnegative");
  }

  ErrorStats stats;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float absolute = std::abs(actual[index] - reference[index]);
    const float denominator = std::max(std::abs(reference[index]), std::numeric_limits<float>::min());
    const float relative = absolute / denominator;
    if (!std::isfinite(absolute) || absolute > stats.max_abs_error) {
      stats.max_abs_error = absolute;
      stats.worst_index = index;
    }
    if (!std::isfinite(relative) || relative > stats.max_rel_error) {
      stats.max_rel_error = relative;
    }
    if (!std::isfinite(actual[index]) || !std::isfinite(reference[index]) ||
        absolute > atol + rtol * std::abs(reference[index])) {
      stats.passed = false;
    }
  }
  return stats;
}

double operation_count(const Problem& problem) {
  validate_problem(problem);
  return 2.0 * problem.m * problem.n * problem.k + 2.0 * problem.m * problem.n;
}

}  // namespace sgemm
