#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sgemm {

struct Problem {
  int m;
  int n;
  int k;
  float alpha;
  float beta;
};

struct RunOptions {
  Problem problem{1024, 1024, 1024, 0.8F, 0.2F};
  int warmup{5};
  int repeat{20};
  std::uint32_t seed{42};
  float atol{-1.0F};
  float rtol{-1.0F};
  bool check{true};
  bool check_only{false};
  bool json{false};
};

struct InputMatrices {
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c0;
};

struct ErrorStats {
  bool passed{true};
  float max_abs_error{0.0F};
  float max_rel_error{0.0F};
  std::size_t worst_index{0};
};

struct BenchmarkResult {
  int method_id{};
  std::string method;
  std::string status;
  double latency_ms{};
  double min_latency_ms{};
  double gflops{};
  ErrorStats errors;
  std::string message;
};

InputMatrices make_inputs(const Problem& problem, std::uint32_t seed);

std::vector<float> cpu_sgemm(const Problem& problem,
                             const std::vector<float>& a,
                             const std::vector<float>& b,
                             const std::vector<float>& c0);

ErrorStats compare_vectors(const std::vector<float>& actual,
                           const std::vector<float>& reference,
                           float atol,
                           float rtol);

double operation_count(const Problem& problem);

}  // namespace sgemm
