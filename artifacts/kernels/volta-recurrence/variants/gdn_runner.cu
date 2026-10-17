// Native KWV-08 runner draft. Compile only after the exclusive task is claimed.
#include "bundle_contract.hpp"
#include "gdn_recurrent_state.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using kwv08::BundleCase;
using kwv08::kAtol;
using kwv08::kRtol;
using kwv08::kOutputSentinel;

std::string json_escape(const std::string& s) {
  std::ostringstream out;
  for (unsigned char c : s) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
        else out << static_cast<char>(c);
    }
  }
  return out.str();
}

std::string cuda_error(cudaError_t status, const char* expression, int line) {
  std::ostringstream out;
  out << "CUDA error at line " << line << " for " << expression << ": "
      << cudaGetErrorString(status);
  return out.str();
}
#define CUDA_CHECK(expr) do { const cudaError_t _status = (expr); if (_status != cudaSuccess) throw std::runtime_error(cuda_error(_status, #expr, __LINE__)); } while (0)

struct CheckResult {
  std::string name;
  std::uint64_t checked = 0, failed = 0, nonfinite = 0;
  double max_abs_error = 0.0, max_scaled_error = 0.0;
  std::string error;
  bool passed() const { return failed == 0 && nonfinite == 0 && error.empty() && checked > 0; }
};

struct TimingResult {
  std::string status = "not_run";
  std::string reason;
  std::vector<double> host_enqueue_us, host_completion_us, device_event_interval_us;
};

struct CaseResult {
  std::string id;
  std::uint32_t T=0, B=0, d_v=0, d_k=0;
  std::string status = "not_run";
  std::string error;
  std::vector<CheckResult> checks;
  TimingResult timing;
};

class DeviceBuffers {
 public:
  float *keys=nullptr, *values=nullptr, *queries=nullptr, *alpha=nullptr, *beta=nullptr;
  float *states=nullptr, *outputs=nullptr;
  int* lengths=nullptr;
  int* segment_lengths=nullptr;
  cudaStream_t stream=nullptr;

  ~DeviceBuffers() {
    // Async copies and kernels all use this explicit stream. Drain it before
    // releasing device storage; run_case declares host vectors before this
    // object so their backing buffers remain alive through this synchronization.
    if (stream) (void)cudaStreamSynchronize(stream);
    if (keys) cudaFree(keys);
    if (values) cudaFree(values);
    if (queries) cudaFree(queries);
    if (alpha) cudaFree(alpha);
    if (beta) cudaFree(beta);
    if (states) cudaFree(states);
    if (outputs) cudaFree(outputs);
    if (lengths) cudaFree(lengths);
    if (segment_lengths) cudaFree(segment_lengths);
    if (stream) cudaStreamDestroy(stream);
  }
  DeviceBuffers(const DeviceBuffers&) = delete;
  DeviceBuffers& operator=(const DeviceBuffers&) = delete;
  DeviceBuffers() = default;
};

void allocate_copy(const BundleCase& c, DeviceBuffers& d,
                   std::vector<float>& host_output) {
  const std::size_t n_k = static_cast<std::size_t>(c.T) * c.B * c.d_k;
  const std::size_t n_v = static_cast<std::size_t>(c.T) * c.B * c.d_v;
  const std::size_t n_gate = static_cast<std::size_t>(c.T) * c.B;
  const std::size_t n_state = static_cast<std::size_t>(c.B) * c.d_v * c.d_k;
  CUDA_CHECK(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
#define ALLOC_AND_COPY(ptr, vec) do { \
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.ptr), sizeof((vec)[0]) * (vec).size())); \
  CUDA_CHECK(cudaMemcpyAsync(d.ptr, (vec).data(), sizeof((vec)[0]) * (vec).size(), cudaMemcpyHostToDevice, d.stream)); \
} while (0)
  (void)n_k; (void)n_v; (void)n_gate; (void)n_state;
  ALLOC_AND_COPY(keys, c.keys);
  ALLOC_AND_COPY(values, c.values);
  ALLOC_AND_COPY(queries, c.queries);
  ALLOC_AND_COPY(alpha, c.alpha);
  ALLOC_AND_COPY(beta, c.beta);
  ALLOC_AND_COPY(lengths, c.lengths);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.segment_lengths), sizeof(int) * c.B));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.states), sizeof(float) * n_state));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d.outputs), sizeof(float) * n_v));
#undef ALLOC_AND_COPY
  host_output.assign(n_v, kOutputSentinel);
  CUDA_CHECK(cudaMemcpyAsync(d.states, c.initial_states.data(), sizeof(float) * n_state,
                             cudaMemcpyHostToDevice, d.stream));
  CUDA_CHECK(cudaMemcpyAsync(d.outputs, host_output.data(), sizeof(float) * n_v,
                             cudaMemcpyHostToDevice, d.stream));
  CUDA_CHECK(cudaStreamSynchronize(d.stream));
}

void reset_state(const BundleCase& c, DeviceBuffers& d, bool reset_output,
                 std::vector<float>& host_output) {
  CUDA_CHECK(cudaMemcpyAsync(d.states, c.initial_states.data(),
                             sizeof(float) * c.initial_states.size(),
                             cudaMemcpyHostToDevice, d.stream));
  if (reset_output) {
    std::fill(host_output.begin(), host_output.end(), kOutputSentinel);
    CUDA_CHECK(cudaMemcpyAsync(d.outputs, host_output.data(),
                               sizeof(float) * host_output.size(),
                               cudaMemcpyHostToDevice, d.stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(d.stream));
}

template <int DV, int DK>
void launch_variant(const std::string& variant,
                    const float* keys, const float* values, const float* queries,
                    const float* alpha, const float* beta, const int* lengths,
                    float* states, float* outputs, int T, int B,
                    cudaStream_t stream) {
  if (variant == "per_token") {
    kwv08::launch_materialized_sequence<DV, DK>(keys, values, queries, alpha, beta,
                                                lengths, states, outputs, T, B, stream);
  } else {
    kwv08::launch_retained_sequence<DV, DK>(keys, values, queries, alpha, beta,
                                            lengths, states, outputs, T, B, stream);
  }
}

void launch_case(const BundleCase& c, const std::string& variant,
                 const float* keys, const float* values, const float* queries,
                 const float* alpha, const float* beta, const int* lengths,
                 float* states, float* outputs, int T, cudaStream_t stream) {
  if (c.d_v == 32 && c.d_k == 32)
    launch_variant<32, 32>(variant, keys, values, queries, alpha, beta, lengths,
                           states, outputs, T, c.B, stream);
  else if (c.d_v == 64 && c.d_k == 64)
    launch_variant<64, 64>(variant, keys, values, queries, alpha, beta, lengths,
                           states, outputs, T, c.B, stream);
  else
    throw std::runtime_error("runner has no compiled specialization for this d_v/d_k");
}

CheckResult check_outputs(const BundleCase& c, const std::vector<float>& actual,
                          const std::string& name) {
  CheckResult result;
  result.name = name;
  const std::size_t expected_size = static_cast<std::size_t>(c.T) * c.B * c.d_v;
  if (actual.size() != expected_size || c.expected_outputs.size() != expected_size) {
    result.error = "output/reference extent mismatch";
    result.failed = 1;
    return result;
  }
  for (std::uint32_t t = 0; t < c.T; ++t) {
    for (std::uint32_t b = 0; b < c.B; ++b) {
      if (t >= static_cast<std::uint32_t>(c.lengths[b])) continue;
      for (std::uint32_t row = 0; row < c.d_v; ++row) {
        const std::size_t i = (static_cast<std::size_t>(t) * c.B + b) * c.d_v + row;
        const double got = actual[i], expected = c.expected_outputs[i];
        ++result.checked;
        if (!std::isfinite(got) || !std::isfinite(expected)) {
          ++result.nonfinite;
          ++result.failed;
          continue;
        }
        const double abs_error = std::abs(got - expected);
        const double allowed = kAtol + kRtol * std::abs(expected);
        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_scaled_error = std::max(result.max_scaled_error, abs_error / allowed);
        if (abs_error > allowed) ++result.failed;
      }
    }
  }
  if (result.checked == 0) result.error = "no live output elements were checked";
  return result;
}

CheckResult check_padding(const BundleCase& c, const std::vector<float>& actual,
                          const std::string& name) {
  CheckResult result;
  result.name = name;
  for (std::uint32_t t = 0; t < c.T; ++t) {
    for (std::uint32_t b = 0; b < c.B; ++b) {
      if (t < static_cast<std::uint32_t>(c.lengths[b])) continue;
      for (std::uint32_t row = 0; row < c.d_v; ++row) {
        const std::size_t i = (static_cast<std::size_t>(t) * c.B + b) * c.d_v + row;
        ++result.checked;
        if (!std::isfinite(actual[i])) {
          ++result.nonfinite;
          ++result.failed;
        } else if (actual[i] != kOutputSentinel) {
          ++result.failed;
        }
      }
    }
  }
  if (result.checked == 0) result.error = "case has no padded outputs to check";
  return result;
}

CheckResult check_final_state(const BundleCase& c, const std::vector<float>& actual,
                              const std::string& name) {
  CheckResult result;
  result.name = name;
  if (actual.size() != c.expected_final_states.size()) {
    result.error = "state/reference extent mismatch";
    result.failed = 1;
    return result;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double got = actual[i], expected = c.expected_final_states[i];
    ++result.checked;
    if (!std::isfinite(got) || !std::isfinite(expected)) {
      ++result.nonfinite;
      ++result.failed;
      continue;
    }
    const double abs_error = std::abs(got - expected);
    const double allowed = kAtol + kRtol * std::abs(expected);
    result.max_abs_error = std::max(result.max_abs_error, abs_error);
    result.max_scaled_error = std::max(result.max_scaled_error, abs_error / allowed);
    if (abs_error > allowed) ++result.failed;
  }
  return result;
}

void run_full(const BundleCase& c, const std::string& variant, DeviceBuffers& d,
              std::vector<float>& host_output, std::vector<float>& host_state,
              CaseResult& result) {
  reset_state(c, d, true, host_output);
  launch_case(c, variant, d.keys, d.values, d.queries, d.alpha, d.beta, d.lengths,
              d.states, d.outputs, static_cast<int>(c.T), d.stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(d.stream));
  CUDA_CHECK(cudaMemcpyAsync(host_output.data(), d.outputs,
                             sizeof(float) * host_output.size(),
                             cudaMemcpyDeviceToHost, d.stream));
  host_state.resize(c.initial_states.size());
  CUDA_CHECK(cudaMemcpyAsync(host_state.data(), d.states,
                             sizeof(float) * host_state.size(),
                             cudaMemcpyDeviceToHost, d.stream));
  CUDA_CHECK(cudaStreamSynchronize(d.stream));
  result.checks.push_back(check_outputs(c, host_output, "full_sequence_all_outputs"));
  result.checks.push_back(check_padding(c, host_output, "full_sequence_padding_untouched"));
  result.checks.push_back(check_final_state(c, host_state, "full_sequence_final_state"));
}

void run_segmented(const BundleCase& c, const std::string& variant, DeviceBuffers& d,
                   std::vector<float>& host_output, std::vector<float>& host_state,
                   CaseResult& result) {
  reset_state(c, d, true, host_output);
  std::vector<std::uint32_t> boundaries{0, std::min(1u, c.T), std::min(33u, c.T),
                                        std::min(129u, c.T), c.T};
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
  std::vector<std::int32_t> lengths(c.B, 0);
  for (std::size_t part = 1; part < boundaries.size(); ++part) {
    const std::uint32_t start = boundaries[part - 1], end = boundaries[part];
    const std::uint32_t segment_T = end - start;
    for (std::uint32_t b = 0; b < c.B; ++b) {
      const std::uint32_t length = static_cast<std::uint32_t>(c.lengths[b]);
      lengths[b] = length > start ? static_cast<std::int32_t>(std::min(length, end) - start) : 0;
    }
    CUDA_CHECK(cudaMemcpyAsync(d.segment_lengths, lengths.data(), sizeof(int) * c.B,
                               cudaMemcpyHostToDevice, d.stream));
    // Reuse the host lengths buffer only after its asynchronous transfer finishes.
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    const std::size_t k_offset = static_cast<std::size_t>(start) * c.B * c.d_k;
    const std::size_t v_offset = static_cast<std::size_t>(start) * c.B * c.d_v;
    const std::size_t gate_offset = static_cast<std::size_t>(start) * c.B;
    const std::size_t out_offset = static_cast<std::size_t>(start) * c.B * c.d_v;
    launch_case(c, variant, d.keys + k_offset, d.values + v_offset,
                d.queries + k_offset, d.alpha + gate_offset, d.beta + gate_offset,
                d.segment_lengths, d.states, d.outputs + out_offset,
                static_cast<int>(segment_T), d.stream);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaStreamSynchronize(d.stream));
  CUDA_CHECK(cudaMemcpyAsync(host_output.data(), d.outputs,
                             sizeof(float) * host_output.size(),
                             cudaMemcpyDeviceToHost, d.stream));
  host_state.resize(c.initial_states.size());
  CUDA_CHECK(cudaMemcpyAsync(host_state.data(), d.states,
                             sizeof(float) * host_state.size(),
                             cudaMemcpyDeviceToHost, d.stream));
  CUDA_CHECK(cudaStreamSynchronize(d.stream));
  result.checks.push_back(check_outputs(c, host_output, "segmented_carry_all_outputs"));
  result.checks.push_back(check_padding(c, host_output, "segmented_padding_untouched"));
  result.checks.push_back(check_final_state(c, host_state, "segmented_carry_final_state"));
}

void time_case(const BundleCase& c, const std::string& variant, DeviceBuffers& d,
               std::vector<float>& host_output, TimingResult& timing) {
  constexpr int kWarmups = 2;
  constexpr int kSamples = 7;
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  try {
    for (int i = 0; i < kWarmups; ++i) {
      reset_state(c, d, false, host_output);
      launch_case(c, variant, d.keys, d.values, d.queries, d.alpha, d.beta,
                  d.lengths, d.states, d.outputs, static_cast<int>(c.T), d.stream);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(d.stream));
    }
    for (int i = 0; i < kSamples; ++i) {
      reset_state(c, d, false, host_output);
      const auto wall_start = Clock::now();
      CUDA_CHECK(cudaEventRecord(start, d.stream));
      launch_case(c, variant, d.keys, d.values, d.queries, d.alpha, d.beta,
                  d.lengths, d.states, d.outputs, static_cast<int>(c.T), d.stream);
      CUDA_CHECK(cudaGetLastError());
      const auto enqueue_done = Clock::now();
      CUDA_CHECK(cudaEventRecord(stop, d.stream));
      CUDA_CHECK(cudaEventSynchronize(stop));
      const auto wall_done = Clock::now();
      float device_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&device_ms, start, stop));
      timing.host_enqueue_us.push_back(
          std::chrono::duration<double, std::micro>(enqueue_done - wall_start).count());
      timing.host_completion_us.push_back(
          std::chrono::duration<double, std::micro>(wall_done - wall_start).count());
      // This stream interval can include device idle time while the host submits
      // many per-token launches; it is not a pure-kernel sum for that variant.
      timing.device_event_interval_us.push_back(static_cast<double>(device_ms) * 1000.0);
    }
    auto valid_samples = [=](const std::vector<double>& values) {
      return values.size() == kSamples && std::all_of(values.begin(), values.end(),
          [](double value) { return std::isfinite(value) && value > 0.0; });
    };
    if (!valid_samples(timing.host_enqueue_us) ||
        !valid_samples(timing.host_completion_us) ||
        !valid_samples(timing.device_event_interval_us))
      throw std::runtime_error("one or more timing samples were missing, non-finite, or non-positive");
    timing.status = "passed";
  } catch (...) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    throw;
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

bool all_checks_passed(const CaseResult& result) {
  return std::all_of(result.checks.begin(), result.checks.end(),
                     [](const CheckResult& c) { return c.passed(); });
}

void run_case(const BundleCase& c, const std::string& variant, const std::string& mode,
              CaseResult& result) {
  result.id = c.id;
  result.T = c.T; result.B = c.B; result.d_v = c.d_v; result.d_k = c.d_k;
  std::vector<float> host_output, host_state;
  // DeviceBuffers is destroyed first. Its destructor drains the explicit
  // stream while these pageable host-copy buffers are still alive.
  DeviceBuffers d;
  allocate_copy(c, d, host_output);
  run_full(c, variant, d, host_output, host_state, result);
  // NCU profile mode is a single, explicitly selected full-sequence launch.
  // The ordinary correctness and timing roles still exercise segmented carry.
  if (mode != "profile")
    run_segmented(c, variant, d, host_output, host_state, result);
  result.status = all_checks_passed(result) ? "passed" : "failed";
  if (mode == "time") {
    if (result.status == "passed") time_case(c, variant, d, host_output, result.timing);
    else {
      result.timing.status = "not_run";
      result.timing.reason = "correctness or segmented-carry checks failed";
    }
  }
}

double median_value(std::vector<double> xs) {
  if (xs.empty()) return 0.0;
  std::sort(xs.begin(), xs.end());
  const std::size_t mid = xs.size() / 2;
  return xs.size() % 2 ? xs[mid] : 0.5 * (xs[mid - 1] + xs[mid]);
}

void print_values(const std::vector<double>& xs) {
  std::cout << '[';
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << xs[i];
  }
  std::cout << ']';
}

void print_check(const CheckResult& c) {
  std::cout << "{\"name\":\"" << json_escape(c.name) << "\",\"status\":\""
            << (c.passed() ? "passed" : "failed") << "\",\"checked\":" << c.checked
            << ",\"failed\":" << c.failed << ",\"nonfinite\":" << c.nonfinite
            << ",\"max_abs_error\":" << c.max_abs_error
            << ",\"max_scaled_error\":" << c.max_scaled_error;
  if (!c.error.empty()) std::cout << ",\"error\":\"" << json_escape(c.error) << '"';
  std::cout << '}';
}

void print_case(const CaseResult& c, const std::string& variant) {
  std::cout << "{\"case_id\":\"" << json_escape(c.id) << "\",\"shape\":{\"T\":"
            << c.T << ",\"B\":" << c.B << ",\"d_v\":" << c.d_v
            << ",\"d_k\":" << c.d_k << "},\"variant\":\""
            << variant << "\",\"status\":\"" << c.status << "\",\"checks\":[";
  for (std::size_t i = 0; i < c.checks.size(); ++i) {
    if (i) std::cout << ',';
    print_check(c.checks[i]);
  }
  std::cout << "],\"timing\":{\"status\":\"" << c.timing.status << '"';
  if (!c.timing.reason.empty())
    std::cout << ",\"reason\":\"" << json_escape(c.timing.reason) << '"';
  if (!c.timing.host_enqueue_us.empty()) {
    std::cout << ",\"host_enqueue_us\":"; print_values(c.timing.host_enqueue_us);
    std::cout << ",\"host_completion_us\":"; print_values(c.timing.host_completion_us);
    std::cout << ",\"device_event_interval_us\":"; print_values(c.timing.device_event_interval_us);
  }
  if (!c.timing.host_enqueue_us.empty()) {
    std::cout << ",\"host_enqueue_median_us\":" << median_value(c.timing.host_enqueue_us)
              << ",\"host_completion_median_us\":" << median_value(c.timing.host_completion_us)
              << ",\"device_event_interval_median_us\":" << median_value(c.timing.device_event_interval_us);
  }
  std::cout << '}';
  if (!c.error.empty()) std::cout << ",\"error\":\"" << json_escape(c.error) << '"';
  std::cout << '}';
}

struct Options { std::string variant, mode, bundle_path, case_id; };
Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 >= argc) throw std::runtime_error("each option requires a value");
    const std::string key = argv[i++], value = argv[i];
    if (key == "--variant") o.variant = value;
    else if (key == "--mode") o.mode = value;
    else if (key == "--bundle") o.bundle_path = value;
    else if (key == "--case") o.case_id = value;
    else throw std::runtime_error("unknown option: " + key);
  }
  if (o.variant != "per_token" && o.variant != "retained")
    throw std::runtime_error("--variant must be per_token or retained");
  if (o.mode != "correctness" && o.mode != "time" && o.mode != "profile")
    throw std::runtime_error("--mode must be correctness, time, or profile");
  if (o.mode == "profile" && o.case_id.empty())
    throw std::runtime_error("profile mode requires an explicit --case");
  if (o.bundle_path.empty()) throw std::runtime_error("--bundle is required");
  return o;
}

void print_failure(const std::string& variant, const std::string& mode,
                   const std::string& message, const std::vector<CaseResult>& results,
                   const std::string& bundle_sha = "", const std::string& file_sha = "",
                   const std::string& gpu = "", const std::string& uuid = "",
                   const std::string& cc = "") {
  std::cout << std::setprecision(12)
            << "{\"format\":\"KERNELWIKI-VOLTA-RECURRENCE-RUN/1\",\"status\":\"failed\","
            << "\"variant\":\"" << json_escape(variant) << "\",\"mode\":\""
            << json_escape(mode) << "\",\"gpu\":\"" << json_escape(gpu)
            << "\",\"gpu_uuid\":\"" << json_escape(uuid)
            << "\",\"compute_capability\":\"" << json_escape(cc)
            << "\",\"bundle_payload_sha256\":\"" << bundle_sha
            << "\",\"bundle_file_sha256\":\"" << file_sha
            << "\",\"tolerance\":{\"atol\":" << kAtol << ",\"rtol\":" << kRtol
            << "},\"error\":\"" << json_escape(message) << "\",\"cases\":[";
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (i) std::cout << ',';
    print_case(results[i], variant);
  }
  std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try { options = parse_args(argc, argv); }
  catch (const std::exception& e) {
    print_failure("", "", e.what(), {});
    return 64;
  }

  kwv08::Bundle bundle;
  try { bundle = kwv08::read_bundle(options.bundle_path); }
  catch (const std::exception& e) {
    print_failure(options.variant, options.mode, e.what(), {});
    return 2;
  }
  if (!options.case_id.empty() && std::none_of(
          bundle.cases.begin(), bundle.cases.end(), [&](const BundleCase& c) {
            return c.id == options.case_id;
          })) {
    print_failure(options.variant, options.mode,
                  "requested case is absent from recurrence bundle: " + options.case_id,
                  {}, bundle.payload_sha256, bundle.file_sha256);
    return 64;
  }

  int device = -1;
  cudaError_t device_status = cudaGetDevice(&device);
  if (device_status != cudaSuccess) {
    std::vector<CaseResult> not_run;
    for (const auto& c : bundle.cases) { CaseResult r; r.id=c.id; r.T=c.T; r.B=c.B; r.d_v=c.d_v; r.d_k=c.d_k; r.error="device unavailable"; not_run.push_back(r); }
    print_failure(options.variant, options.mode,
                  cuda_error(device_status, "cudaGetDevice", __LINE__), not_run,
                  bundle.payload_sha256, bundle.file_sha256);
    return 3;
  }
  cudaDeviceProp properties{};
  device_status = cudaGetDeviceProperties(&properties, device);
  if (device_status != cudaSuccess) {
    std::vector<CaseResult> not_run;
    for (const auto& c : bundle.cases) { CaseResult r; r.id=c.id; r.T=c.T; r.B=c.B; r.d_v=c.d_v; r.d_k=c.d_k; r.error="device properties unavailable"; not_run.push_back(r); }
    print_failure(options.variant, options.mode,
                  cuda_error(device_status, "cudaGetDeviceProperties", __LINE__), not_run,
                  bundle.payload_sha256, bundle.file_sha256);
    return 3;
  }
  std::ostringstream uuid_stream;
  uuid_stream << std::hex << std::setfill('0');
  for (const auto byte : properties.uuid.bytes)
    uuid_stream << std::setw(2) << (static_cast<unsigned>(static_cast<unsigned char>(byte)) & 0xffu);
  const std::string uuid = uuid_stream.str();
  const std::string cc = std::to_string(properties.major) + "." + std::to_string(properties.minor);
  const std::string gpu = properties.name;
  if (properties.major != 7 || properties.minor != 0) {
    std::vector<CaseResult> not_run;
    for (const auto& c : bundle.cases) { CaseResult r; r.id=c.id; r.T=c.T; r.B=c.B; r.d_v=c.d_v; r.d_k=c.d_k; r.error="native runner requires SM70"; not_run.push_back(r); }
    print_failure(options.variant, options.mode, "expected compute capability 7.0",
                  not_run, bundle.payload_sha256, bundle.file_sha256, gpu, uuid, cc);
    return 3;
  }

  std::vector<CaseResult> results;
  results.reserve(bundle.cases.size());
  bool all_passed = true;
  for (const auto& c : bundle.cases) {
    if (!options.case_id.empty() && c.id != options.case_id) continue;
    CaseResult result;
    result.id = c.id;
    try { run_case(c, options.variant, options.mode, result); }
    catch (const std::exception& e) { result.status="failed"; result.error=e.what(); }
    all_passed = all_passed && result.status == "passed" &&
        (options.mode != "time" || result.timing.status == "passed");
    results.push_back(std::move(result));
  }

  std::cout << std::setprecision(12)
            << "{\"format\":\"KERNELWIKI-VOLTA-RECURRENCE-RUN/1\",\"status\":\""
            << (all_passed ? "passed" : "failed") << "\",\"variant\":\""
            << options.variant << "\",\"mode\":\"" << options.mode
            << "\",\"gpu\":\"" << json_escape(gpu) << "\",\"gpu_uuid\":\""
            << uuid << "\",\"compute_capability\":\"" << cc
            << "\",\"bundle_payload_sha256\":\"" << bundle.payload_sha256
            << "\",\"bundle_file_sha256\":\"" << bundle.file_sha256
            << "\",\"tolerance\":{\"atol\":" << kAtol << ",\"rtol\":" << kRtol
            << "},\"timing_semantics\":{\"host_enqueue_us\":\"wall time through submission of all launches\","
            << "\"host_completion_us\":\"wall time through stream completion\","
            << "\"device_event_interval_us\":\"CUDA event interval; can include device idle while host submits per-token launches\"},"
            << "\"cases\":[";
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (i) std::cout << ',';
    print_case(results[i], options.variant);
  }
  std::cout << "]}\n";
  return all_passed ? 0 : 5;
}
