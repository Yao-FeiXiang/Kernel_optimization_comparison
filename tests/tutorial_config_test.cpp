#include "sgemm/tutorial_config.hpp"

using namespace sgemm::tutorial;

static_assert(kKernel5Fast == TileConfig{128, 128, 8, 8, 8});
static_assert(kKernel9Candidates.size() == 6);
static_assert(kKernel9Candidates[0] == TileConfig{64, 64, 16, 4, 4});
static_assert(kKernel9Candidates[5] == TileConfig{128, 128, 16, 8, 8});
static_assert(kKernel10A100.block_m == 64);
static_assert(kKernel10A100.block_n == 128);
static_assert(kKernel10A100.warp_m == 32);
static_assert(kKernel10A100.warp_n == 64);
static_assert(kKernel10A100.warp_n_iterations == 1);
static_assert(kKernel10A100.thread_m == 4);
static_assert(kKernel10A100.thread_n == 4);
static_assert(kKernel10A100.threads == 128);
static_assert(kKernel10Generic.block_m == 128);
static_assert(kKernel10Generic.warp_m == 64);
static_assert(kKernel10Generic.warp_n_iterations == 4);

int main() {}
