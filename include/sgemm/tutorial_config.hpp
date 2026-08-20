#pragma once

#include <array>

namespace sgemm::tutorial {

struct TileConfig {
  int block_m;
  int block_n;
  int block_k;
  int thread_m;
  int thread_n;
};

constexpr bool operator==(const TileConfig& left, const TileConfig& right) {
  return left.block_m == right.block_m && left.block_n == right.block_n &&
         left.block_k == right.block_k && left.thread_m == right.thread_m &&
         left.thread_n == right.thread_n;
}

struct WarpConfig {
  int block_m;
  int block_n;
  int block_k;
  int warp_m;
  int warp_n;
  int warp_n_iterations;
  int thread_m;
  int thread_n;
  int threads;
};

inline constexpr TileConfig kKernel5Fast{128, 128, 8, 8, 8};

inline constexpr std::array<TileConfig, 6> kKernel9Candidates{{
    {64, 64, 16, 4, 4},
    {64, 64, 8, 8, 8},
    {128, 64, 8, 8, 8},
    {64, 128, 8, 8, 8},
    {128, 128, 8, 8, 8},
    {128, 128, 16, 8, 8},
}};

inline constexpr WarpConfig kKernel10A100{
    64, 128, 16, 32, 64, 1, 4, 4, 128,
};

inline constexpr WarpConfig kKernel10Generic{
    128, 128, 16, 64, 64, 4, 8, 4, 128,
};

}  // namespace sgemm::tutorial
