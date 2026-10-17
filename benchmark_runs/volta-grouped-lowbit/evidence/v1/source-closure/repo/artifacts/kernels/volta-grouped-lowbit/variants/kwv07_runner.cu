#include "include/kwv07_bundle.hpp"
#include "grouped_lowbit.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kWarmups = 2;
constexpr int kSamples = 9;
constexpr double kUnitRoundoff = 0x1p-24;
constexpr double kSiluLipschitz = 1.0 + 0.3678794411714423216;  // 1 + 1/e, conservative.

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

void cublas_check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(status));
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ == 0) throw std::invalid_argument("zero-sized device allocation");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return count_; }
 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

class Stream {
 public:
  Stream() { cuda_check(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking), "cudaStreamCreate"); }
  ~Stream() { if (value_) cudaStreamDestroy(value_); }
  operator cudaStream_t() const { return value_; }
 private:
  cudaStream_t value_ = nullptr;
};

class BlasHandle {
 public:
  explicit BlasHandle(cudaStream_t stream) {
    cublas_check(cublasCreate(&value_), "cublasCreate");
    cublas_check(cublasSetStream(value_, stream), "cublasSetStream");
    cublas_check(cublasSetMathMode(value_, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
  }
  ~BlasHandle() { if (value_) cublasDestroy(value_); }
  operator cublasHandle_t() const { return value_; }
 private:
  cublasHandle_t value_ = nullptr;
};

struct Timing {
  std::vector<double> event_us;
  std::vector<double> host_us;
  double median_event_us = 0.0;
  double median_host_us = 0.0;
};

struct ReuseTiming {
  int uses = 0;
  Timing timing;
};

struct Check {
  std::size_t checked = 0;
  std::size_t failed = 0;
  std::size_t nonfinite = 0;
  double max_abs_error = 0.0;
  double max_bound = 0.0;
  double max_ratio = 0.0;
};

struct CaseResult {
  std::string id;
  std::size_t checked = 0;
  std::size_t failed = 0;
  std::size_t nonfinite = 0;
  double max_abs_error = 0.0;
  double max_bound = 0.0;
  double max_ratio = 0.0;
  std::vector<int> reuse;
  std::vector<Timing> timings;
  std::vector<ReuseTiming> reuse_timings;
  double prep_event_us = 0.0;
  double prep_host_us = 0.0;
  double metadata_host_us = 0.0;
  double setup_host_us = 0.0;
};

struct Options {
  std::string bundle;
  std::string suite;
  std::string implementation;
  std::string mode = "check";
  std::string only_case;
};

Options parse(int argc, char** argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto take = [&](const char* name) {
      if (++i >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
      return std::string(argv[i]);
    };
    if (arg == "--bundle") result.bundle = take("--bundle");
    else if (arg == "--suite") result.suite = take("--suite");
    else if (arg == "--impl") result.implementation = take("--impl");
    else if (arg == "--mode") result.mode = take("--mode");
    else if (arg == "--case") result.only_case = take("--case");
    else if (arg == "--help") {
      std::cout << "usage: kwv07_runner --bundle DIR --suite grouped|w4a16 "
                   "--impl split-gated|paired-gated|predecode-w4|direct-w4 "
                   "[--mode check|time] [--case ID]\n";
      std::exit(0);
    } else throw std::invalid_argument("unknown argument: " + arg);
  }
  if (result.bundle.empty() || result.suite.empty() || result.implementation.empty())
    throw std::invalid_argument("--bundle, --suite, and --impl are required");
  if (result.mode != "check" && result.mode != "time")
    throw std::invalid_argument("--mode must be check or time");
  const bool grouped = result.suite == "grouped";
  const bool lowbit = result.suite == "w4a16";
  if (!grouped && !lowbit) throw std::invalid_argument("--suite must be grouped or w4a16");
  if (grouped && result.implementation != "split-gated" && result.implementation != "paired-gated")
    throw std::invalid_argument("grouped suite requires split-gated or paired-gated");
  if (lowbit && result.implementation != "predecode-w4" && result.implementation != "direct-w4")
    throw std::invalid_argument("w4a16 suite requires predecode-w4 or direct-w4");
  return result;
}

std::vector<std::string> case_ids(const Options& options) {
  std::vector<std::string> cases = options.suite == "grouped"
      ? std::vector<std::string>{"tiny_balanced", "tiny_skewed", "moderate_balanced", "moderate_skewed",
                                 "weighted_balanced", "weighted_skewed"}
      : std::vector<std::string>{"low_m", "moderate_m", "weighted_low_m", "weighted_moderate_m"};
  if (!options.only_case.empty()) {
    if (std::find(cases.begin(), cases.end(), options.only_case) == cases.end())
      throw std::invalid_argument("--case does not belong to selected suite");
    return {options.only_case};
  }
  return cases;
}

std::vector<std::uint32_t> stable_group_order(const kwv07::Input& input,
                                             std::vector<std::uint32_t>& offsets,
                                             double& prep_us) {
  const auto start = Clock::now();
  const auto& h = input.header;
  std::vector<std::vector<std::uint32_t>> buckets(h.experts);
  for (std::uint32_t index = 0; index < input.assignments.size(); ++index)
    buckets[static_cast<std::size_t>(input.assignments[index].expert_id)].push_back(index);
  offsets.assign(h.experts + 1, 0);
  std::vector<std::uint32_t> order;
  order.reserve(input.assignments.size());
  for (std::uint32_t expert = 0; expert < h.experts; ++expert) {
    offsets[expert] = static_cast<std::uint32_t>(order.size());
    order.insert(order.end(), buckets[expert].begin(), buckets[expert].end());
  }
  offsets[h.experts] = static_cast<std::uint32_t>(order.size());
  if (order.size() != input.assignments.size())
    throw std::runtime_error("stable expert grouping lost assignments");
  prep_us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
  return order;
}

__global__ void gather_assigned_rows(const std::uint16_t* x, const kwv_grouped::Assignment* assignments,
                                     const std::uint32_t* group_order, std::uint16_t* packed_x,
                                     int assignment_count, int k) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(assignment_count) * k;
  if (index >= total) return;
  const int grouped_row = static_cast<int>(index / k);
  const int kk = static_cast<int>(index % k);
  const auto assignment_index = group_order[grouped_row];
  const auto row = assignments[assignment_index].row_id;
  packed_x[index] = x[static_cast<std::size_t>(row) * k + kk];
}

__global__ void gated_epilogue_scatter(const float* up, const float* gate,
                                       const kwv_grouped::Assignment* assignments,
                                       const std::uint32_t* group_order, float* output,
                                       int assignment_count, int n) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(assignment_count) * n;
  if (index >= total) return;
  const int grouped_row = static_cast<int>(index / n);
  const int col = static_cast<int>(index % n);
  const auto assignment_index = group_order[grouped_row];
  const int output_position = assignments[assignment_index].output_position;
  const std::size_t grouped_offset = index;
  const float g = gate[grouped_offset];
  const float silu = g / (1.0f + expf(-g));
  output[static_cast<std::size_t>(output_position) * n + col] = silu * up[grouped_offset];
}

void launch_cublas_rowmajor(cublasHandle_t handle, const std::uint16_t* a_bits,
                            const std::uint16_t* b_bits, float* c,
                            int m, int n, int k) {
  // Row-major C[M,N] = A[M,K] * B[K,N] is column-major C^T = B^T * A^T.
  const float alpha = 1.0f, beta = 0.0f;
  cublas_check(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, m, k, &alpha,
                            b_bits, CUDA_R_16F, n,
                            a_bits, CUDA_R_16F, k,
                            &beta, c, CUDA_R_32F, n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "cublasGemmEx FP16/FP32");
}

struct GroupedBuffers {
  double prep_us = 0.0;
  std::vector<std::uint32_t> offsets;
  std::vector<std::uint32_t> order;
  std::vector<kwv_grouped::Assignment> assignments;
  DeviceBuffer<std::uint16_t> x;
  DeviceBuffer<std::uint16_t> up;
  DeviceBuffer<std::uint16_t> gate;
  DeviceBuffer<kwv_grouped::Assignment> d_assignments;
  DeviceBuffer<std::uint32_t> d_order;
  DeviceBuffer<std::uint16_t> packed_x;
  DeviceBuffer<float> grouped_up;
  DeviceBuffer<float> grouped_gate;
  DeviceBuffer<float> output;

  explicit GroupedBuffers(const kwv07::Input& input, cudaStream_t stream)
      : order(stable_group_order(input, offsets, prep_us)),
        x(input.x_or_activation_bits.size()), up(input.up_bits.size()), gate(input.gate_bits.size()),
        d_assignments(input.assignments.size()), d_order(order.size()),
        packed_x(input.assignments.size() * input.header.k),
        grouped_up(input.assignments.size() * input.header.n),
        grouped_gate(input.assignments.size() * input.header.n),
        output(input.assignments.size() * input.header.n) {
    assignments.reserve(input.assignments.size());
    for (const auto& item : input.assignments)
      assignments.push_back({item.row_id, item.expert_id, item.output_position});
    cuda_check(cudaMemcpyAsync(x.get(), input.x_or_activation_bits.data(), x.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream), "copy grouped X");
    cuda_check(cudaMemcpyAsync(up.get(), input.up_bits.data(), up.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream), "copy grouped W_up");
    cuda_check(cudaMemcpyAsync(gate.get(), input.gate_bits.data(), gate.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream), "copy grouped W_gate");
    cuda_check(cudaMemcpyAsync(d_assignments.get(), assignments.data(), assignments.size() * sizeof(assignments[0]), cudaMemcpyHostToDevice, stream), "copy grouped assignments");
    cuda_check(cudaMemcpyAsync(d_order.get(), order.data(), order.size() * sizeof(order[0]), cudaMemcpyHostToDevice, stream), "copy group order");
  }
};

void run_grouped_split(const kwv07::Input& input, GroupedBuffers& b,
                       cublasHandle_t handle, cudaStream_t stream) {
  const auto& h = input.header;
  constexpr int Threads = 256;
  const std::size_t gather_count = input.assignments.size() * h.k;
  gather_assigned_rows<<<(gather_count + Threads - 1) / Threads, Threads, 0, stream>>>(
      b.x.get(), b.d_assignments.get(), b.d_order.get(), b.packed_x.get(),
      static_cast<int>(input.assignments.size()), static_cast<int>(h.k));
  cuda_check(cudaGetLastError(), "gather assigned rows");
  for (std::uint32_t expert = 0; expert < h.experts; ++expert) {
    const std::uint32_t begin = b.offsets[expert];
    const std::uint32_t end = b.offsets[expert + 1];
    const int count = static_cast<int>(end - begin);
    if (count == 0) continue;
    const std::size_t weight_offset = static_cast<std::size_t>(expert) * h.k * h.n;
    const std::size_t row_offset = static_cast<std::size_t>(begin) * h.k;
    const std::size_t output_offset = static_cast<std::size_t>(begin) * h.n;
    launch_cublas_rowmajor(handle, b.packed_x.get() + row_offset,
                           b.up.get() + weight_offset,
                           b.grouped_up.get() + output_offset,
                           count, static_cast<int>(h.n), static_cast<int>(h.k));
    launch_cublas_rowmajor(handle, b.packed_x.get() + row_offset,
                           b.gate.get() + weight_offset,
                           b.grouped_gate.get() + output_offset,
                           count, static_cast<int>(h.n), static_cast<int>(h.k));
  }
  const std::size_t output_count = input.assignments.size() * h.n;
  gated_epilogue_scatter<<<(output_count + Threads - 1) / Threads, Threads, 0, stream>>>(
      b.grouped_up.get(), b.grouped_gate.get(), b.d_assignments.get(), b.d_order.get(),
      b.output.get(), static_cast<int>(input.assignments.size()), static_cast<int>(h.n));
  cuda_check(cudaGetLastError(), "grouped gated epilogue scatter");
}

void run_grouped_paired(const kwv07::Input& input, GroupedBuffers& b,
                        cudaStream_t stream) {
  const auto& h = input.header;
  cuda_check(kwv_grouped::launch_gated_pair_small_m(
      b.x.get(), b.up.get(), b.gate.get(), b.d_assignments.get(), b.output.get(),
      static_cast<int>(h.rows_or_m), static_cast<int>(h.experts),
      static_cast<int>(h.assignment_count), static_cast<int>(h.k), static_cast<int>(h.n), stream),
      "paired gated small-M candidate");
}

void launch_dense(cublasHandle_t handle, const std::uint16_t* activation,
                  const std::uint16_t* weights, float* output,
                  int m, int n, int k) {
  launch_cublas_rowmajor(handle, activation, weights, output, m, n, k);
}

struct W4Buffers {
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<std::uint8_t> packed;
  DeviceBuffer<float> scales;
  std::unique_ptr<DeviceBuffer<std::uint16_t>> decoded;
  DeviceBuffer<float> output;

  W4Buffers(const kwv07::Input& input, bool needs_decoded, cudaStream_t stream)
      : activation(input.x_or_activation_bits.size()), packed(input.packed_codes.size()),
        scales(input.scales.size()),
        output(static_cast<std::size_t>(input.header.rows_or_m) * input.header.n) {
    if (needs_decoded)
      decoded = std::make_unique<DeviceBuffer<std::uint16_t>>(
          static_cast<std::size_t>(input.header.k) * input.header.n);
    cuda_check(cudaMemcpyAsync(activation.get(), input.x_or_activation_bits.data(), activation.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream), "copy W4 activations");
    cuda_check(cudaMemcpyAsync(packed.get(), input.packed_codes.data(), packed.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice, stream), "copy W4 codes");
    cuda_check(cudaMemcpyAsync(scales.get(), input.scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy W4 scales");
  }
};

void decode_w4(W4Buffers& b, const kwv07::Input& input, cudaStream_t stream) {
  if (!b.decoded) throw std::runtime_error("predecode buffer was not allocated");
  const auto& h = input.header;
  cuda_check(kwv_grouped::launch_decode_w4a16(
      b.packed.get(), b.scales.get(), b.decoded->get(), static_cast<int>(h.k),
      static_cast<int>(h.n), static_cast<int>(h.group_size), stream), "decode W4A16 weights");
}

void run_w4_predecode(W4Buffers& b, const kwv07::Input& input,
                      cublasHandle_t handle, cudaStream_t stream) {
  decode_w4(b, input, stream);
  const auto& h = input.header;
  launch_dense(handle, b.activation.get(), b.decoded->get(), b.output.get(),
               static_cast<int>(h.rows_or_m), static_cast<int>(h.n), static_cast<int>(h.k));
}

void run_w4_direct(W4Buffers& b, const kwv07::Input& input, cudaStream_t stream) {
  const auto& h = input.header;
  cuda_check(kwv_grouped::launch_w4a16_simt_small_m(
      b.activation.get(), b.packed.get(), b.scales.get(), b.output.get(),
      static_cast<int>(h.rows_or_m), static_cast<int>(h.k), static_cast<int>(h.n),
      static_cast<int>(h.group_size), stream), "direct W4A16 small-M candidate");
}

double gamma_bound(std::uint32_t k, double sum_abs) {
  const double ku = static_cast<double>(k) * kUnitRoundoff;
  if (!(ku < 1.0)) throw std::runtime_error("K invalidates the FP32 gamma bound");
  return (ku / (1.0 - ku)) * sum_abs;
}

double silu64(double value) {
  if (value >= 0.0) return value / (1.0 + std::exp(-value));
  const double e = std::exp(value);
  return value * e / (1.0 + e);
}

Check check_output(const kwv07::Input& input, const kwv07::Oracle& oracle,
                   const std::vector<float>& actual) {
  kwv07::require_matching(input, oracle);
  const std::size_t count = oracle.output.size();
  if (actual.size() != count) throw std::runtime_error("native output extent differs from oracle");
  Check result;
  const bool grouped = input.header.kind == kwv07::Kind::Grouped;
  for (std::size_t i = 0; i < count; ++i) {
    ++result.checked;
    const double reference = oracle.output[i];
    double bound = 0.0;
    if (grouped) {
      const double u_ref = oracle.up_reference[i];
      const double g_ref = oracle.gate_reference[i];
      const double e_u = gamma_bound(input.header.k, oracle.sum_abs_up[i]);
      const double e_g = gamma_bound(input.header.k, oracle.sum_abs_gate[i]);
      const double activation = silu64(g_ref);
      bound = std::abs(activation) * e_u +
              (std::abs(u_ref) + e_u) * kSiluLipschitz * e_g +
              8.0 * kUnitRoundoff * (1.0 + std::abs(reference) + std::abs(u_ref) + std::abs(g_ref));
    } else {
      bound = gamma_bound(input.header.k, oracle.sum_abs_products[i]);
    }
    result.max_bound = std::max(result.max_bound, bound);
    if (!std::isfinite(actual[i])) {
      ++result.nonfinite;
      ++result.failed;
      continue;
    }
    const double error = std::abs(static_cast<double>(actual[i]) - reference);
    result.max_abs_error = std::max(result.max_abs_error, error);
    const double ratio = bound == 0.0 ? (error == 0.0 ? 0.0 : std::numeric_limits<double>::max())
                                      : error / bound;
    result.max_ratio = std::max(result.max_ratio, ratio);
    if (error > bound) ++result.failed;
  }
  return result;
}

Timing measure(const std::function<void()>& invoke, cudaStream_t stream) {
  Timing result;
  for (int i = 0; i < kWarmups; ++i) invoke();
  cuda_check(cudaStreamSynchronize(stream), "sync benchmark warmups");
  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "create timing start event");
  cuda_check(cudaEventCreate(&stop), "create timing stop event");
  for (int i = 0; i < kSamples; ++i) {
    const auto host_start = Clock::now();
    cuda_check(cudaEventRecord(start, stream), "record timing start");
    invoke();
    cuda_check(cudaEventRecord(stop, stream), "record timing stop");
    cuda_check(cudaEventSynchronize(stop), "synchronize timing sample");
    const auto host_stop = Clock::now();
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "read timing events");
    result.event_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
    result.host_us.push_back(std::chrono::duration<double, std::micro>(host_stop - host_start).count());
  }
  cuda_check(cudaEventDestroy(stop), "destroy timing stop");
  cuda_check(cudaEventDestroy(start), "destroy timing start");
  auto median = [](std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  };
  result.median_event_us = median(result.event_us);
  result.median_host_us = median(result.host_us);
  return result;
}

Timing measure_repeated(const std::function<void()>& invoke, int uses,
                        cudaStream_t stream) {
  if (uses <= 0) throw std::invalid_argument("reuse count must be positive");
  return measure([&]() {
    for (int i = 0; i < uses; ++i) invoke();
  }, stream);
}

std::vector<float> copy_output(const DeviceBuffer<float>& output, cudaStream_t stream) {
  std::vector<float> host(output.size());
  cuda_check(cudaMemcpyAsync(host.data(), output.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy native output");
  cuda_check(cudaStreamSynchronize(stream), "sync correctness output");
  return host;
}

CaseResult run_grouped_case(const Options& options, const std::string& id,
                            cudaStream_t stream, cublasHandle_t handle) {
  const auto setup_start = Clock::now();
  const std::string input_path = options.bundle + "/inputs/" + id + ".bin";
  const std::string oracle_path = options.bundle + "/oracles/" + id + ".bin";
  const auto input = kwv07::read_input(input_path);
  const auto oracle = kwv07::read_oracle(oracle_path);
  if (input.header.kind != kwv07::Kind::Grouped) throw std::runtime_error("grouped suite got W4 case");
  GroupedBuffers buffers(input, stream);
  cuda_check(cudaStreamSynchronize(stream), "complete grouped setup transfer");
  const double setup_host_us = std::chrono::duration<double, std::micro>(Clock::now() - setup_start).count();
  auto invoke = [&]() {
    if (options.implementation == "split-gated") run_grouped_split(input, buffers, handle, stream);
    else run_grouped_paired(input, buffers, stream);
  };
  invoke();
  auto actual = copy_output(buffers.output, stream);
  const auto check = check_output(input, oracle, actual);
  CaseResult result{id, check.checked, check.failed, check.nonfinite,
                    check.max_abs_error, check.max_bound, check.max_ratio};
  result.setup_host_us = setup_host_us;
  result.metadata_host_us = buffers.prep_us;
  if (options.mode == "time") {
    result.timings.push_back(measure(invoke, stream));
    // Check again after timing so the final native PASS follows every timed output.
    invoke();
    const auto post_timing = check_output(input, oracle, copy_output(buffers.output, stream));
    result.checked = post_timing.checked;
    result.failed = post_timing.failed;
    result.nonfinite = post_timing.nonfinite;
    result.max_abs_error = post_timing.max_abs_error;
    result.max_bound = post_timing.max_bound;
    result.max_ratio = post_timing.max_ratio;
  }
  return result;
}

CaseResult run_w4_case(const Options& options, const std::string& id,
                       cudaStream_t stream, cublasHandle_t handle) {
  const auto setup_start = Clock::now();
  const std::string input_path = options.bundle + "/inputs/" + id + ".bin";
  const std::string oracle_path = options.bundle + "/oracles/" + id + ".bin";
  const auto input = kwv07::read_input(input_path);
  const auto oracle = kwv07::read_oracle(oracle_path);
  if (input.header.kind != kwv07::Kind::W4A16) throw std::runtime_error("W4 suite got grouped case");
  W4Buffers buffers(input, options.implementation == "predecode-w4", stream);
  cuda_check(cudaStreamSynchronize(stream), "complete W4 setup transfer");
  const double setup_host_us = std::chrono::duration<double, std::micro>(Clock::now() - setup_start).count();
  auto invoke = [&]() {
    if (options.implementation == "predecode-w4") run_w4_predecode(buffers, input, handle, stream);
    else run_w4_direct(buffers, input, stream);
  };
  invoke();
  auto actual = copy_output(buffers.output, stream);
  const auto check = check_output(input, oracle, actual);
  CaseResult result{id, check.checked, check.failed, check.nonfinite,
                    check.max_abs_error, check.max_bound, check.max_ratio};
  result.setup_host_us = setup_host_us;
  if (options.mode == "time") {
    std::function<void()> compute;
    if (options.implementation == "predecode-w4") {
      auto prep = measure([&]() { decode_w4(buffers, input, stream); }, stream);
      result.prep_event_us = prep.median_event_us;
      result.prep_host_us = prep.median_host_us;
      // Decode once after the sampled preparation and keep those exact weights
      // resident for all reuse-count GEMMs.
      decode_w4(buffers, input, stream);
      cuda_check(cudaStreamSynchronize(stream), "sync final W4 decode");
      compute = [&]() {
        const auto& h = input.header;
        launch_dense(handle, buffers.activation.get(), buffers.decoded->get(), buffers.output.get(),
                     static_cast<int>(h.rows_or_m), static_cast<int>(h.n), static_cast<int>(h.k));
      };
    } else {
      compute = [&]() { run_w4_direct(buffers, input, stream); };
    }
    const bool low_m = id == "low_m" || id == "weighted_low_m";
    result.reuse = low_m ? std::vector<int>{1, 4, 16} : std::vector<int>{1, 4};
    for (const int uses : result.reuse) {
      ReuseTiming row;
      row.uses = uses;
      row.timing = measure_repeated(compute, uses, stream);
      if (uses == 1) result.timings.push_back(row.timing);
      result.reuse_timings.push_back(std::move(row));
    }
    // Recheck after preparation/reuse timing before emitting PASS metadata.
    invoke();
    const auto post_timing = check_output(input, oracle, copy_output(buffers.output, stream));
    result.checked = post_timing.checked;
    result.failed = post_timing.failed;
    result.nonfinite = post_timing.nonfinite;
    result.max_abs_error = post_timing.max_abs_error;
    result.max_bound = post_timing.max_bound;
    result.max_ratio = post_timing.max_ratio;
  }
  return result;
}

void json_string(const std::string& value) {
  std::cout << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') std::cout << '\\' << static_cast<char>(c);
    else if (c < 0x20) std::cout << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                                  << static_cast<unsigned>(c) << std::dec;
    else std::cout << static_cast<char>(c);
  }
  std::cout << '"';
}

void json_array(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << values[i];
  }
  std::cout << ']';
}

void emit_result(const Options& options, const cudaDeviceProp& properties,
                 const std::string& raw_uuid, const std::vector<CaseResult>& cases) {
  std::size_t case_count = cases.size(), checked = 0, failed = 0, nonfinite = 0;
  double max_abs = 0.0, max_bound = 0.0, max_ratio = 0.0;
  for (const auto& item : cases) {
    checked += item.checked;
    failed += item.failed;
    nonfinite += item.nonfinite;
    max_abs = std::max(max_abs, item.max_abs_error);
    max_bound = std::max(max_bound, item.max_bound);
    max_ratio = std::max(max_ratio, item.max_ratio);
  }
  const bool passed = failed == 0 && nonfinite == 0 && case_count != 0;
  std::cout << std::setprecision(17)
            << "{\"format\":\"KERNELWIKI-KWV07-NATIVE/1\",\"status\":\""
            << (passed ? "passed" : "failed") << "\",\"suite\":";
  json_string(options.suite);
  std::cout << ",\"implementation\":";
  json_string(options.implementation);
  std::cout << ",\"mode\":";
  json_string(options.mode);
  std::cout << ",\"gpu\":";
  json_string(properties.name);
  std::cout << ",\"gpu_uuid\":";
  json_string(raw_uuid);
  std::cout << ",\"compute_capability\":\"" << properties.major << '.' << properties.minor
            << "\",\"case_count\":" << case_count << ",\"checked_outputs\":" << checked
            << ",\"failed_outputs\":" << failed << ",\"nonfinite_outputs\":" << nonfinite
            << ",\"max_abs_error\":" << max_abs << ",\"max_admission_tolerance\":" << max_bound
            << ",\"max_tolerance_ratio\":" << max_ratio << ",\"cases\":[";
  for (std::size_t i = 0; i < cases.size(); ++i) {
    if (i) std::cout << ',';
    const auto& item = cases[i];
    std::cout << "{\"case_id\":";
    json_string(item.id);
    std::cout << ",\"checked_outputs\":" << item.checked << ",\"failed_outputs\":" << item.failed
              << ",\"nonfinite_outputs\":" << item.nonfinite << ",\"max_abs_error\":" << item.max_abs_error
              << ",\"max_admission_tolerance\":" << item.max_bound
              << ",\"max_tolerance_ratio\":" << item.max_ratio
              << ",\"one_time_setup_host_us\":" << item.setup_host_us
              << ",\"group_metadata_host_us\":" << item.metadata_host_us
              << ",\"predecode_event_us\":" << item.prep_event_us
              << ",\"predecode_host_us\":" << item.prep_host_us;
    if (!item.timings.empty()) {
      const auto& timing = item.timings.front();
      std::cout << ",\"event_samples_us\":";
      json_array(timing.event_us);
      std::cout << ",\"host_samples_us\":";
      json_array(timing.host_us);
      std::cout << ",\"median_event_us\":" << timing.median_event_us
                << ",\"median_host_us\":" << timing.median_host_us;
      if (options.suite == "grouped")
        std::cout << ",\"cold_one_use_host_total_us\":"
                  << item.setup_host_us + timing.median_host_us;
      if (!item.reuse.empty()) {
        std::cout << ",\"reuse_assumption\":\"represented weights, scales, and activation remain resident and unchanged for each reuse count; cold setup includes one file read, allocation, and H2D transfer\",\"reuse_runs\":[";
        for (std::size_t j = 0; j < item.reuse_timings.size(); ++j) {
          if (j) std::cout << ',';
          const auto& reuse = item.reuse_timings[j];
          std::cout << "{\"uses\":" << reuse.uses
                    << ",\"device_event_samples_us\":";
          json_array(reuse.timing.event_us);
          std::cout << ",\"compute_host_samples_us\":";
          json_array(reuse.timing.host_us);
          std::cout << ",\"cold_device_event_total_us\":"
                    << (options.implementation == "predecode-w4" ? item.prep_event_us : 0.0) +
                       reuse.timing.median_event_us
                    << ",\"cold_host_wall_total_us\":"
                    << item.setup_host_us + item.prep_host_us + reuse.timing.median_host_us
                    << '}';
        }
        std::cout << ']';
      }
    }
    std::cout << '}';
  }
  std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse(argc, argv);
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (properties.major != 7 || properties.minor != 0)
      throw std::runtime_error("runner requires native SM70; no fallback path is defined");
    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    for (const auto byte : properties.uuid.bytes)
      uuid << std::setw(2) << (static_cast<unsigned>(static_cast<unsigned char>(byte)) & 0xffu);

    Stream stream;
    BlasHandle handle(stream);
    std::vector<CaseResult> results;
    for (const auto& id : case_ids(options)) {
      if (options.suite == "grouped")
        results.push_back(run_grouped_case(options, id, stream, handle));
      else
        results.push_back(run_w4_case(options, id, stream, handle));
    }
    emit_result(options, properties, uuid.str(), results);
    const bool passed = std::all_of(results.begin(), results.end(), [](const CaseResult& r) {
      return r.failed == 0 && r.nonfinite == 0 && r.checked != 0;
    });
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "kwv07_runner: " << error.what() << '\n';
    return 2;
  }
}
