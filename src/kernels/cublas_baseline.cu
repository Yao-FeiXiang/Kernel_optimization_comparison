#include "sgemm/kernels.cuh"

#ifndef SGEMM_ENABLE_CUBLAS
#define SGEMM_ENABLE_CUBLAS 0
#endif

#if SGEMM_ENABLE_CUBLAS
#include <cublas_v2.h>
#endif

#include <string>

namespace sgemm {
namespace {

#if SGEMM_ENABLE_CUBLAS
cublasHandle_t g_handle = nullptr;

std::string cublas_error(cublasStatus_t status) {
  const char* text = cublasGetStatusString(status);
  return text == nullptr ? "unknown cuBLAS error" : text;
}
#endif

}  // namespace

PrepareResult prepare_cublas(const Problem&, const DeviceOperands& operands) {
#if SGEMM_ENABLE_CUBLAS
  if (g_handle == nullptr) {
    const cublasStatus_t create_status = cublasCreate(&g_handle);
    if (create_status != CUBLAS_STATUS_SUCCESS) {
      return {false, true, {}, "cublasCreate: " + cublas_error(create_status)};
    }
  }
  const cublasStatus_t stream_status = cublasSetStream(g_handle, operands.stream);
  if (stream_status != CUBLAS_STATUS_SUCCESS) {
    return {false, true, {}, "cublasSetStream: " + cublas_error(stream_status)};
  }
  return {true, true, "cuBLAS", {}};
#else
  static_cast<void>(operands);
  return {true, false, {}, "cuBLAS library was not available at build time"};
#endif
}

LaunchResult launch_cublas(const Problem& problem, const DeviceOperands& operands) {
#if SGEMM_ENABLE_CUBLAS
  const cublasStatus_t status = cublasSgemm(
      g_handle, CUBLAS_OP_N, CUBLAS_OP_N, problem.n, problem.m, problem.k, &problem.alpha,
      operands.b, problem.n, operands.a, problem.k, &problem.beta, operands.c, problem.n);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return {false, "cublasSgemm: " + cublas_error(status), {}};
  }
  return {true, {}, "cublas-sgemm"};
#else
  static_cast<void>(problem);
  static_cast<void>(operands);
  return {false, "cuBLAS library was not available at build time", {}};
#endif
}

void cleanup_cublas() {
#if SGEMM_ENABLE_CUBLAS
  if (g_handle != nullptr) {
    cublasDestroy(g_handle);
    g_handle = nullptr;
  }
#endif
}

}  // namespace sgemm
