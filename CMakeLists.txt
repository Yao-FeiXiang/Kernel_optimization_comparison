cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

project(cuda_sgemm_teaching LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(sgemm_reference STATIC src/reference.cpp)
target_include_directories(sgemm_reference PUBLIC include)

enable_testing()
add_executable(reference_test tests/reference_test.cpp)
target_link_libraries(reference_test PRIVATE sgemm_reference)
add_test(NAME reference_test COMMAND reference_test)

add_library(sgemm_kernels STATIC
  src/kernels/registry.cu
  src/kernels/kernel_01_naive.cu
  src/kernels/kernel_02_coalesced.cu
  src/kernels/kernel_03_shared.cu
  src/kernels/kernel_04_blocktiling_1d.cu
  src/kernels/kernel_05_blocktiling_2d.cu
  src/kernels/kernel_06_vectorized.cu
)
target_include_directories(sgemm_kernels PUBLIC include)

add_executable(registry_test tests/registry_test.cu)
target_link_libraries(registry_test PRIVATE sgemm_kernels)
add_test(NAME registry_test COMMAND registry_test)

add_executable(sgemm_bench src/benchmark.cu)
target_link_libraries(sgemm_bench PRIVATE sgemm_reference sgemm_kernels)
