#include "sgemm/benchmark.hpp"
#include "sgemm/kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    check_cuda(cudaMalloc(&pointer_, elements * sizeof(float)), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  float* get() { return pointer_; }
  const float* get() const { return pointer_; }
  std::size_t bytes() const { return elements_ * sizeof(float); }

  static void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
  }

 private:
  float* pointer_{nullptr};
  std::size_t elements_{};
};

class Stream {
 public:
  Stream() { DeviceBuffer::check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
  ~Stream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_{nullptr};
};

class Event {
 public:
  Event() { DeviceBuffer::check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~Event() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{nullptr};
};

struct ParsedArguments {
  sgemm::RunOptions options;
  std::optional<std::string> kernel;
  bool all{false};
  bool list{false};
  bool help{false};
};

struct DeviceInfo {
  std::string name;
  std::string compute_capability;
  std::string cuda_runtime;
};

std::string help_text() {
  return R"(Usage: sgemm_bench (--kernel ID_OR_NAME | --all | --list) [options]

Selectors:
  --kernel VALUE   Run one method by tutorial ID or stable name
  --all            Run every registered method
  --list           List methods without opening a CUDA device

Problem:
  --m INT          Rows of A and C (default: 1024)
  --n INT          Columns of B and C (default: 1024)
  --k INT          Columns of A / rows of B (default: 1024)
  --alpha FLOAT    SGEMM alpha (default: 0.8)
  --beta FLOAT     SGEMM beta (default: 0.2)

Measurement:
  --warmup INT     Warmup launches (default: 5)
  --repeat INT     Timed launches (default: 20)
  --seed INT       Input seed (default: 42)
  --check          Validate against the CPU reference before timing
  --check-only     Validate and skip performance timing
  --atol FLOAT     Absolute tolerance (default: 1e-4*K)
  --rtol FLOAT     Relative tolerance (default: 1e-4)
  --json           Emit one JSON object per method in addition to text
  --help           Show this help
)";
}

std::string require_value(int argc, char** argv, int& index, const std::string& option) {
  if (index + 1 >= argc) {
    throw UsageError(option + " requires a value");
  }
  ++index;
  return argv[index];
}

int parse_int(const std::string& text, const std::string& option) {
  std::size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &consumed);
  } catch (const std::exception&) {
    throw UsageError(option + " requires an integer");
  }
  if (consumed != text.size() || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw UsageError(option + " requires an integer in range");
  }
  return static_cast<int>(value);
}

float parse_float(const std::string& text, const std::string& option) {
  std::size_t consumed = 0;
  float value = 0.0F;
  try {
    value = std::stof(text, &consumed);
  } catch (const std::exception&) {
    throw UsageError(option + " requires a floating-point value");
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    throw UsageError(option + " requires a finite floating-point value");
  }
  return value;
}

ParsedArguments parse_arguments(int argc, char** argv) {
  ParsedArguments parsed;
  parsed.options.check = false;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help" || option == "-h") {
      parsed.help = true;
    } else if (option == "--kernel") {
      parsed.kernel = require_value(argc, argv, index, option);
    } else if (option == "--all") {
      parsed.all = true;
    } else if (option == "--list") {
      parsed.list = true;
    } else if (option == "--m") {
      parsed.options.problem.m = parse_int(require_value(argc, argv, index, option), option);
    } else if (option == "--n") {
      parsed.options.problem.n = parse_int(require_value(argc, argv, index, option), option);
    } else if (option == "--k") {
      parsed.options.problem.k = parse_int(require_value(argc, argv, index, option), option);
    } else if (option == "--alpha") {
      parsed.options.problem.alpha = parse_float(require_value(argc, argv, index, option), option);
    } else if (option == "--beta") {
      parsed.options.problem.beta = parse_float(require_value(argc, argv, index, option), option);
    } else if (option == "--warmup") {
      parsed.options.warmup = parse_int(require_value(argc, argv, index, option), option);
    } else if (option == "--repeat") {
      parsed.options.repeat = parse_int(require_value(argc, argv, index, option), option);
    } else if (option == "--seed") {
      const int seed = parse_int(require_value(argc, argv, index, option), option);
      if (seed < 0) {
        throw UsageError("--seed must be nonnegative");
      }
      parsed.options.seed = static_cast<std::uint32_t>(seed);
    } else if (option == "--atol") {
      parsed.options.atol = parse_float(require_value(argc, argv, index, option), option);
    } else if (option == "--rtol") {
      parsed.options.rtol = parse_float(require_value(argc, argv, index, option), option);
    } else if (option == "--check") {
      parsed.options.check = true;
    } else if (option == "--check-only") {
      parsed.options.check = true;
      parsed.options.check_only = true;
    } else if (option == "--json") {
      parsed.options.json = true;
    } else {
      throw UsageError("unknown option: " + option);
    }
  }

  if (parsed.kernel && parsed.all) {
    throw UsageError("--kernel and --all are mutually exclusive");
  }
  const auto& problem = parsed.options.problem;
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0) {
    throw UsageError("matrix dimensions must be positive");
  }
  if (parsed.options.warmup < 0) {
    throw UsageError("--warmup must be nonnegative");
  }
  if (parsed.options.repeat <= 0) {
    throw UsageError("--repeat must be positive");
  }
  if (parsed.options.atol < 0.0F && parsed.options.atol != -1.0F) {
    throw UsageError("--atol must be nonnegative");
  }
  if (parsed.options.rtol < 0.0F && parsed.options.rtol != -1.0F) {
    throw UsageError("--rtol must be nonnegative");
  }
  if (parsed.kernel && !sgemm::find_kernel(*parsed.kernel)) {
    throw UsageError("unknown kernel: " + *parsed.kernel);
  }
  if (!parsed.help && !parsed.list && !parsed.kernel && !parsed.all) {
    throw UsageError("specify --kernel, --all, or --list");
  }
  return parsed;
}

DeviceInfo query_device() {
  int count = 0;
  DeviceBuffer::check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (count == 0) {
    throw std::runtime_error("no CUDA device is available");
  }
  DeviceBuffer::check_cuda(cudaSetDevice(0), "cudaSetDevice");
  cudaDeviceProp properties{};
  DeviceBuffer::check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
  int runtime_version = 0;
  DeviceBuffer::check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  return {
      properties.name,
      std::to_string(properties.major) + "." + std::to_string(properties.minor),
      std::to_string(runtime_version / 1000) + "." +
          std::to_string((runtime_version % 1000) / 10),
  };
}

void copy_async(float* destination,
                const float* source,
                std::size_t bytes,
                cudaMemcpyKind kind,
                cudaStream_t stream,
                std::string_view operation) {
  DeviceBuffer::check_cuda(cudaMemcpyAsync(destination, source, bytes, kind, stream), operation);
}

void restore_c(DeviceBuffer& c, const DeviceBuffer& c0, cudaStream_t stream) {
  copy_async(c.get(), c0.get(), c.bytes(), cudaMemcpyDeviceToDevice, stream, "restore C0");
}

sgemm::BenchmarkResult run_method(const sgemm::KernelSpec& kernel,
                                  const sgemm::RunOptions& options,
                                  const std::vector<float>& reference,
                                  DeviceBuffer& a,
                                  DeviceBuffer& b,
                                  const DeviceBuffer& c0,
                                  DeviceBuffer& c,
                                  cudaStream_t stream) {
  sgemm::BenchmarkResult result;
  result.method_id = kernel.id;
  result.method = std::string(kernel.name);
  result.status = "pass";

  if (options.check) {
    restore_c(c, c0, stream);
    const cudaError_t launch_status = kernel.launch(options.problem, a.get(), b.get(), c.get(), stream);
    DeviceBuffer::check_cuda(launch_status, std::string(kernel.name) + " validation launch");
    std::vector<float> actual(reference.size());
    copy_async(actual.data(), c.get(), c.bytes(), cudaMemcpyDeviceToHost, stream, "copy validation C");
    DeviceBuffer::check_cuda(cudaStreamSynchronize(stream), "validation synchronize");
    const float atol = options.atol >= 0.0F ? options.atol : 1.0e-4F * options.problem.k;
    const float rtol = options.rtol >= 0.0F ? options.rtol : 1.0e-4F;
    result.errors = sgemm::compare_vectors(actual, reference, atol, rtol);
    if (!result.errors.passed) {
      result.status = "fail";
      std::ostringstream message;
      message << "correctness mismatch at index " << result.errors.worst_index;
      result.message = message.str();
      return result;
    }
  }

  if (options.check_only) {
    return result;
  }

  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    restore_c(c, c0, stream);
    DeviceBuffer::check_cuda(
        kernel.launch(options.problem, a.get(), b.get(), c.get(), stream),
        std::string(kernel.name) + " warmup launch");
  }
  DeviceBuffer::check_cuda(cudaStreamSynchronize(stream), "warmup synchronize");

  Event start;
  Event stop;
  std::vector<float> samples;
  samples.reserve(static_cast<std::size_t>(options.repeat));
  for (int iteration = 0; iteration < options.repeat; ++iteration) {
    restore_c(c, c0, stream);
    DeviceBuffer::check_cuda(cudaEventRecord(start.get(), stream), "record start event");
    DeviceBuffer::check_cuda(
        kernel.launch(options.problem, a.get(), b.get(), c.get(), stream),
        std::string(kernel.name) + " timed launch");
    DeviceBuffer::check_cuda(cudaEventRecord(stop.get(), stream), "record stop event");
    DeviceBuffer::check_cuda(cudaEventSynchronize(stop.get()), "synchronize stop event");
    float milliseconds = 0.0F;
    DeviceBuffer::check_cuda(
        cudaEventElapsedTime(&milliseconds, start.get(), stop.get()), "measure elapsed time");
    samples.push_back(milliseconds);
  }
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  result.latency_ms = samples.size() % 2 == 0
                          ? 0.5 * (samples[middle - 1] + samples[middle])
                          : samples[middle];
  result.min_latency_ms = samples.front();
  result.gflops = sgemm::operation_count(options.problem) / (result.latency_ms * 1.0e6);
  return result;
}

std::string json_escape(std::string_view value) {
  std::ostringstream output;
  for (const char character : value) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default: output << character;
    }
  }
  return output.str();
}

std::string timestamp_utc() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

void print_human(const sgemm::BenchmarkResult& result) {
  std::cout << "[" << result.method_id << "] " << result.method << ": " << result.status;
  if (result.status == "pass" && result.latency_ms > 0.0) {
    std::cout << ", median " << std::fixed << std::setprecision(3) << result.latency_ms
              << " ms, " << result.gflops << " GFLOP/s";
  }
  if (!result.message.empty()) {
    std::cout << ", " << result.message;
  }
  std::cout << '\n';
}

void print_json(const sgemm::BenchmarkResult& result,
                const sgemm::RunOptions& options,
                const DeviceInfo& device,
                const std::string& timestamp) {
  const auto& problem = options.problem;
  std::cout << std::setprecision(9)
            << "{\"schema_version\":1"
            << ",\"method_id\":" << result.method_id
            << ",\"method\":\"" << json_escape(result.method) << "\""
            << ",\"status\":\"" << json_escape(result.status) << "\""
            << ",\"m\":" << problem.m << ",\"n\":" << problem.n << ",\"k\":" << problem.k
            << ",\"alpha\":" << problem.alpha << ",\"beta\":" << problem.beta
            << ",\"warmup\":" << options.warmup << ",\"repeat\":" << options.repeat
            << ",\"latency_ms\":" << result.latency_ms
            << ",\"min_latency_ms\":" << result.min_latency_ms
            << ",\"gflops\":" << result.gflops
            << ",\"max_abs_error\":" << result.errors.max_abs_error
            << ",\"max_rel_error\":" << result.errors.max_rel_error
            << ",\"gpu\":\"" << json_escape(device.name) << "\""
            << ",\"compute_capability\":\"" << json_escape(device.compute_capability) << "\""
            << ",\"cuda_runtime\":\"" << json_escape(device.cuda_runtime) << "\""
            << ",\"timestamp_utc\":\"" << timestamp << "\""
            << ",\"message\":\"" << json_escape(result.message) << "\"}\n";
}

int run(const ParsedArguments& parsed) {
  if (parsed.help) {
    std::cout << help_text();
    return 0;
  }
  if (parsed.list) {
    for (const auto& kernel : sgemm::phase_a_kernels()) {
      std::cout << kernel.id << ' ' << kernel.name << " - " << kernel.description << '\n';
    }
    return 0;
  }

  std::vector<sgemm::KernelSpec> kernels;
  if (parsed.all) {
    kernels = sgemm::phase_a_kernels();
  } else {
    kernels.push_back(*sgemm::find_kernel(*parsed.kernel));
  }

  const DeviceInfo device = query_device();
  const auto inputs = sgemm::make_inputs(parsed.options.problem, parsed.options.seed);
  std::vector<float> reference;
  if (parsed.options.check) {
    reference = sgemm::cpu_sgemm(parsed.options.problem, inputs.a, inputs.b, inputs.c0);
  }
  DeviceBuffer a(inputs.a.size());
  DeviceBuffer b(inputs.b.size());
  DeviceBuffer c0(inputs.c0.size());
  DeviceBuffer c(inputs.c0.size());
  Stream stream;
  copy_async(a.get(), inputs.a.data(), a.bytes(), cudaMemcpyHostToDevice, stream.get(), "copy A");
  copy_async(b.get(), inputs.b.data(), b.bytes(), cudaMemcpyHostToDevice, stream.get(), "copy B");
  copy_async(c0.get(), inputs.c0.data(), c0.bytes(), cudaMemcpyHostToDevice, stream.get(), "copy C0");
  DeviceBuffer::check_cuda(cudaStreamSynchronize(stream.get()), "input copy synchronize");

  bool failed = false;
  const std::string timestamp = timestamp_utc();
  for (const auto& kernel : kernels) {
    sgemm::BenchmarkResult result;
    try {
      result = run_method(kernel, parsed.options, reference, a, b, c0, c, stream.get());
    } catch (const std::exception& error) {
      result.method_id = kernel.id;
      result.method = std::string(kernel.name);
      result.status = "fail";
      result.message = error.what();
    }
    failed = failed || result.status != "pass";
    print_human(result);
    if (parsed.options.json) {
      print_json(result, parsed.options, device, timestamp);
    }
  }
  return failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_arguments(argc, argv));
  } catch (const UsageError& error) {
    std::cerr << "Usage error: " << error.what() << "\n\n" << help_text();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
