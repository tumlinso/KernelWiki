// Native correctness/timing runner for the fixed KWV-05 row suite.
#include "row_kernels.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kInputMagic[] = "KWVROWS1";
constexpr char kReferenceMagic[] = "KWVRREF1";
constexpr std::size_t kCaseIdBytes = 32;
constexpr float kPaddingSentinel = -123456.75f;
constexpr int kMaxCols = 8193;
constexpr float kAtol = 0x1p-30f;
constexpr float kRtol = 0x1p-14f;

struct Case {
  std::string id;
  bool softmax = false;
  int rows = 0, cols = 0;
  int x_stride = 0, y_stride = 0, mask_stride = 0;
  int x_offset = 0, y_offset = 0, mask_offset = 0, gamma_offset = 0;
  std::uint32_t flags = 0;
  float epsilon = 0.0f, atol = 0.0f, rtol = 0.0f;
  std::vector<float> x, gamma;
  std::vector<std::uint8_t> mask;
  std::vector<double> reference;
};

struct Options {
  std::string input, reference, implementation, mode = "correctness";
  std::vector<std::string> case_ids;
  int warmups = 5, repetitions = 21, rotations = 1;
};

void require_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

std::vector<std::string> split_case_ids(const std::string& value) {
  std::vector<std::string> ids;
  std::size_t start = 0;
  while (start < value.size()) {
    const std::size_t comma = value.find(',', start);
    const std::size_t end = comma == std::string::npos ? value.size() : comma;
    if (end == start) throw std::invalid_argument("--case contains an empty ID");
    ids.push_back(value.substr(start, end - start));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (ids.empty() || value.back() == ',')
    throw std::invalid_argument("--case requires one or more nonempty IDs");
  return ids;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* name) {
      if (++i >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
      return std::string(argv[i]);
    };
    if (arg == "--input") o.input = value("--input");
    else if (arg == "--reference") o.reference = value("--reference");
    else if (arg == "--impl") o.implementation = value("--impl");
    else if (arg == "--mode") o.mode = value("--mode");
    else if (arg == "--case") o.case_ids = split_case_ids(value("--case"));
    else if (arg == "--warmups") o.warmups = std::stoi(value("--warmups"));
    else if (arg == "--repetitions") o.repetitions = std::stoi(value("--repetitions"));
    else if (arg == "--rotations") o.rotations = std::stoi(value("--rotations"));
    else throw std::invalid_argument("unknown argument: " + arg);
  }
  if (o.input.empty() || o.reference.empty() || o.implementation.empty())
    throw std::invalid_argument("--input, --reference, and --impl are required");
  if (o.mode != "correctness" && o.mode != "time")
    throw std::invalid_argument("--mode must be correctness or time");
  if (o.implementation != "baseline" && o.implementation != "candidate" &&
      o.implementation != "warp" && o.implementation != "cta" &&
      o.implementation != "vec4")
    throw std::invalid_argument("--impl must be baseline, candidate, warp, cta, or vec4");
  if (o.warmups < 0 || o.repetitions <= 0 || o.rotations <= 0 || o.rotations > 16)
    throw std::invalid_argument("invalid warmups, repetitions, or rotations");
  if (o.mode == "time" && o.case_ids.empty())
    throw std::invalid_argument("timing requires --case ID[,ID...]");
  return o;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  in.seekg(0, std::ios::end);
  const auto size = in.tellg();
  if (size < 0) throw std::runtime_error("cannot determine file length: " + path);
  in.seekg(0);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!bytes.empty() && !in.read(reinterpret_cast<char*>(bytes.data()), size))
    throw std::runtime_error("cannot read complete file: " + path);
  return bytes;
}

class Cursor {
 public:
  explicit Cursor(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}
  void magic(const char* expected, std::size_t length) {
    need(length);
    if (std::memcmp(bytes_.data() + offset_, expected, length) != 0)
      throw std::runtime_error("binary file magic mismatch");
    offset_ += length;
  }
  std::uint32_t u32() {
    need(4);
    const auto* p = bytes_.data() + offset_;
    offset_ += 4;
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
           (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
  }
  int dim() {
    const std::uint32_t value = u32();
    if (value > static_cast<std::uint32_t>(INT_MAX))
      throw std::runtime_error("row dimension/stride/offset exceeds INT_MAX");
    return static_cast<int>(value);
  }
  std::uint64_t u64() {
    const std::uint64_t lo = u32();
    const std::uint64_t hi = u32();
    return lo | (hi << 32);
  }
  float f32() {
    const std::uint32_t bits = u32();
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  double f64() {
    const std::uint64_t bits = u64();
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  std::string fixed_id() {
    need(kCaseIdBytes);
    const char* p = reinterpret_cast<const char*>(bytes_.data() + offset_);
    std::size_t n = 0;
    while (n < kCaseIdBytes && p[n] != '\0') ++n;
    if (n == 0 || n == kCaseIdBytes)
      throw std::runtime_error("invalid fixed-width case ID");
    for (std::size_t i = n; i < kCaseIdBytes; ++i)
      if (p[i] != '\0') throw std::runtime_error("case ID padding is not zero");
    offset_ += kCaseIdBytes;
    return std::string(p, n);
  }
  std::vector<float> floats(std::uint32_t count) {
    if (count > (bytes_.size() - offset_) / 4) throw std::runtime_error("truncated FP32 array");
    std::vector<float> values(count);
    for (auto& value : values) value = f32();
    return values;
  }
  std::vector<std::uint8_t> bytes(std::uint32_t count) {
    need(count);
    std::vector<std::uint8_t> values(bytes_.begin() + offset_, bytes_.begin() + offset_ + count);
    offset_ += count;
    return values;
  }
  bool done() const { return offset_ == bytes_.size(); }
 private:
  void need(std::size_t count) const {
    if (count > bytes_.size() - offset_) throw std::runtime_error("truncated binary file");
  }
  const std::vector<std::uint8_t>& bytes_;
  std::size_t offset_ = 0;
};

std::vector<Case> read_cases(const std::string& path) {
  const auto bytes = read_file(path);
  Cursor c(bytes);
  c.magic(kInputMagic, sizeof(kInputMagic) - 1);
  const std::uint32_t count = c.u32();
  if (count == 0 || count > 10000) throw std::runtime_error("invalid row case count");
  std::vector<Case> cases;
  std::map<std::string, bool> ids;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (c.u32() != 1) throw std::runtime_error("unsupported row case version");
    Case x;
    x.id = c.fixed_id();
    const std::uint32_t op = c.u32();
    x.rows = c.dim();
    x.cols = c.dim();
    x.x_stride = c.dim();
    x.y_stride = c.dim();
    x.mask_stride = c.dim();
    x.x_offset = c.dim();
    x.y_offset = c.dim();
    x.mask_offset = c.dim();
    x.gamma_offset = c.dim();
    const std::uint32_t x_count = c.u32();
    const std::uint32_t mask_count = c.u32();
    const std::uint32_t gamma_count = c.u32();
    x.flags = c.u32();
    x.epsilon = c.f32(); x.atol = c.f32(); x.rtol = c.f32();
    if (op != 1 && op != 2) throw std::runtime_error("invalid operation code");
    x.softmax = op == 1;
    if (x.flags & ~1u) throw std::runtime_error("unsupported row flags");
    x.x = c.floats(x_count);
    x.mask = c.bytes(mask_count);
    x.gamma = c.floats(gamma_count);
    if (ids.emplace(x.id, true).second == false)
      throw std::runtime_error("duplicate row case ID");
    const std::int64_t x_need = x.rows ? std::int64_t(x.x_offset) +
        std::int64_t(x.rows - 1) * x.x_stride + x.cols : 0;
    const std::int64_t y_need = x.rows ? std::int64_t(x.y_offset) +
        std::int64_t(x.rows - 1) * x.y_stride + x.cols : 0;
    const std::int64_t mask_need = x.rows && (x.flags & 1u) ?
        std::int64_t(x.mask_offset) + std::int64_t(x.rows - 1) * x.mask_stride + x.cols : 0;
    const std::int64_t gamma_need = x.softmax ? 0 : std::int64_t(x.gamma_offset) + x.cols;
    if (x.cols < 1 || x.cols > kMaxCols || x.x_stride < x.cols || x.y_stride < x.cols ||
        x_count != x_need || mask_count != mask_need || gamma_count != gamma_need ||
        x_need > INT_MAX || y_need > INT_MAX || mask_need > INT_MAX || gamma_need > INT_MAX ||
        ((x.flags & 1u) && x.mask_stride < x.cols) ||
        (!(x.flags & 1u) && (x.mask_offset || mask_count)) ||
        (x.flags & ~1u) || x.atol != kAtol || x.rtol != kRtol ||
        (x.softmax && (x.gamma_offset || x.epsilon != 0.0f)) ||
        (!x.softmax && (x.flags & 1u || !std::isfinite(x.epsilon) ||
                        x.epsilon < 1.0e-5f || x.epsilon > 1.0e-2f)) ||
        !std::isfinite(x.atol) || !std::isfinite(x.rtol) || x.atol < 0.0f || x.rtol < 0.0f)
      throw std::runtime_error("row case extents or numerical contract are invalid: " + x.id);
    if (x.rows == 0 && (x.x_offset || x.y_offset || x.mask_offset || x_count || mask_count))
      throw std::runtime_error("zero-row case must have empty input and zero offsets: " + x.id);
    for (int row = 0; row < x.rows; ++row) {
      for (int col = 0; col < x.cols; ++col) {
        const std::size_t index = std::size_t(x.x_offset) + std::size_t(row) * x.x_stride + col;
        const float value = x.x[index];
        const float bound = x.softmax ? 16.0f : 64.0f;
        if (!std::isfinite(value) || std::abs(value) > bound)
          throw std::runtime_error("logical x outside finite declared domain: " + x.id);
        if (x.flags & 1u) {
          const auto bit = x.mask[x.mask_offset + row * x.mask_stride + col];
          if (bit > 1) throw std::runtime_error("mask byte is not 0/1: " + x.id);
        }
      }
    }
    for (int col = 0; col < x.cols && !x.softmax; ++col) {
      const float value = x.gamma[x.gamma_offset + col];
      if (!std::isfinite(value) || std::abs(value) > 4.0f)
        throw std::runtime_error("logical gamma outside finite declared domain: " + x.id);
    }
    cases.push_back(std::move(x));
  }
  if (!c.done()) throw std::runtime_error("row input bundle has trailing bytes");
  return cases;
}

void read_references(const std::string& path, std::vector<Case>& cases) {
  const auto bytes = read_file(path);
  Cursor c(bytes);
  c.magic(kReferenceMagic, sizeof(kReferenceMagic) - 1);
  const std::uint32_t count = c.u32();
  if (count != cases.size()) throw std::runtime_error("reference case count differs from input");
  std::map<std::string, bool> seen;
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::string id = c.fixed_id();
    const std::uint64_t values = c.u64();
    auto found = std::find_if(cases.begin(), cases.end(), [&](const Case& x) { return x.id == id; });
    if (found == cases.end() || seen.emplace(id, true).second == false ||
        values != std::uint64_t(found->rows) * found->cols)
      throw std::runtime_error("reference case ID or logical extent differs: " + id);
    if (values > (bytes.size() / sizeof(double)))
      throw std::runtime_error("reference value count is unreasonable");
    found->reference.reserve(static_cast<std::size_t>(values));
    for (std::uint64_t j = 0; j < values; ++j) {
      const double value = c.f64();
      if (!std::isfinite(value)) throw std::runtime_error("non-finite CPU reference: " + id);
      found->reference.push_back(value);
    }
  }
  if (!c.done()) throw std::runtime_error("reference bundle has trailing bytes");
}

class DeviceFloats {
 public:
  explicit DeviceFloats(std::size_t count) : count_(std::max<std::size_t>(count, 1)) {
    require_cuda(cudaMalloc(&data_, count_ * sizeof(float)), "cudaMalloc float buffer");
  }
  ~DeviceFloats() { if (data_) cudaFree(data_); }
  DeviceFloats(const DeviceFloats&) = delete;
  DeviceFloats& operator=(const DeviceFloats&) = delete;
  float* get() { return data_; }
 private:
  float* data_ = nullptr;
  std::size_t count_ = 0;
};

class DeviceBytes {
 public:
  explicit DeviceBytes(std::size_t count) : count_(std::max<std::size_t>(count, 1)) {
    require_cuda(cudaMalloc(&data_, count_), "cudaMalloc byte buffer");
  }
  ~DeviceBytes() { if (data_) cudaFree(data_); }
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;
  std::uint8_t* get() { return data_; }
 private:
  std::uint8_t* data_ = nullptr;
  std::size_t count_ = 0;
};

std::size_t output_extent(const Case& c) {
  return c.rows ? std::size_t(c.y_offset) + std::size_t(c.rows - 1) * c.y_stride + c.cols : 0;
}

std::string effective_implementation(const Case& c, const std::string& implementation) {
  if (implementation == "baseline") return c.softmax ? "warp" : "cta";
  if (implementation == "candidate") return c.softmax ? "cta" : "vec4";
  return implementation;
}

std::string implementation_path(const Case& c, const std::string& implementation,
                                kwv_rows::RmsnormCtaPath selected_path) {
  if (c.rows == 0) return "no_launch";
  const std::string route = effective_implementation(c, implementation);
  if (c.softmax) return route == "warp" ? "softmax_warp" : "softmax_cta";
  if (route == "warp") return "rmsnorm_warp";
  if (route != "vec4") return "rmsnorm_cta_scalar";
  return selected_path == kwv_rows::RmsnormCtaPath::kVec4 ? "vec4" : "scalar_fallback";
}

void launch(const Case& c, const std::string& implementation,
            float* x, std::uint8_t* mask, float* gamma, float* y,
            cudaStream_t stream, kwv_rows::RmsnormCtaPath* selected_path) {
  if (c.rows == 0) {
    if (selected_path) *selected_path = kwv_rows::RmsnormCtaPath::kNoLaunch;
    return;
  }
  const std::string route = effective_implementation(c, implementation);
  constexpr int kThreads = 128;
  if (c.softmax) {
    if (route == "vec4") throw std::invalid_argument("vec4 is RMSNorm-only");
    if (route == "warp") {
      const int warps_per_cta = kThreads / kwv_rows::kWarp;
      const int blocks = (c.rows + warps_per_cta - 1) / warps_per_cta;
      kwv_rows::softmax_warp<<<blocks, kThreads, 0, stream>>>(
          x + c.x_offset, (c.flags & 1u) ? mask + c.mask_offset : nullptr,
          y + c.y_offset, c.rows, c.cols, c.x_stride, c.mask_stride, c.y_stride);
    } else {
      kwv_rows::softmax_cta<<<c.rows, kThreads, 0, stream>>>(
          x + c.x_offset, (c.flags & 1u) ? mask + c.mask_offset : nullptr,
          y + c.y_offset, c.rows, c.cols, c.x_stride, c.mask_stride, c.y_stride);
    }
    if (selected_path) *selected_path = kwv_rows::RmsnormCtaPath::kNoLaunch;
  } else if (route == "warp") {
    const int warps_per_cta = kThreads / kwv_rows::kWarp;
    const int blocks = (c.rows + warps_per_cta - 1) / warps_per_cta;
    kwv_rows::rmsnorm_warp<<<blocks, kThreads, 0, stream>>>(
        x + c.x_offset, gamma + c.gamma_offset, y + c.y_offset,
        c.rows, c.cols, c.x_stride, c.y_stride, c.epsilon);
    if (selected_path) *selected_path = kwv_rows::RmsnormCtaPath::kNoLaunch;
  } else {
    const cudaError_t status = kwv_rows::launch_rmsnorm_cta(
        x + c.x_offset, gamma + c.gamma_offset, y + c.y_offset,
        c.rows, c.cols, c.x_stride, c.y_stride, c.epsilon,
        route == "vec4", stream, selected_path);
    require_cuda(status, "RMSNorm CTA launch");
  }
  require_cuda(cudaGetLastError(), "row kernel launch");
}

struct CaseCheck {
  std::uint64_t checked = 0, failed = 0, nonfinite = 0;
  std::uint64_t padding_checked = 0, padding_writes = 0;
  double max_error = 0.0, max_tolerance = 0.0, max_ratio = 0.0;
  std::string selected_path = "not_applicable";
};

CaseCheck execute_correctness(Case& c, const std::string& implementation,
                              cudaStream_t stream) {
  DeviceFloats dx(c.x.size()), dg(c.gamma.size());
  DeviceBytes dm(c.mask.size());
  const std::size_t y_count = output_extent(c);
  DeviceFloats dy(y_count);
  // Keep initialization, kernel work, and readback ordered on this nonblocking
  // stream. Pageable HtoD cudaMemcpy may return after staging while DMA is still
  // in flight, and its default-stream ordering does not order this stream.
  if (!c.x.empty()) require_cuda(cudaMemcpyAsync(dx.get(), c.x.data(), c.x.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "queue represented x");
  if (!c.gamma.empty()) require_cuda(cudaMemcpyAsync(dg.get(), c.gamma.data(), c.gamma.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "queue represented gamma");
  if (!c.mask.empty()) require_cuda(cudaMemcpyAsync(dm.get(), c.mask.data(), c.mask.size(), cudaMemcpyHostToDevice, stream), "queue represented mask");
  std::vector<float> output(std::max<std::size_t>(y_count, 1), kPaddingSentinel);
  require_cuda(cudaMemcpyAsync(dy.get(), output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "queue output/padding sentinels");
  kwv_rows::RmsnormCtaPath selected = kwv_rows::RmsnormCtaPath::kNoLaunch;
  launch(c, implementation, dx.get(), dm.get(), dg.get(), dy.get(), stream, &selected);
  require_cuda(cudaMemcpyAsync(output.data(), dy.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream), "queue correctness output");
  require_cuda(cudaStreamSynchronize(stream), "synchronize correctness run and readback");
  CaseCheck check;
  check.selected_path = implementation_path(c, implementation, selected);

  for (int row = 0; row < c.rows; ++row) {
    for (int col = 0; col < c.cols; ++col) {
      const std::size_t index = std::size_t(c.y_offset) + std::size_t(row) * c.y_stride + col;
      const std::size_t ref_index = std::size_t(row) * c.cols + col;
      const double actual = output[index];
      const double reference = c.reference[ref_index];
      ++check.checked;
      if (!std::isfinite(actual)) {
        ++check.nonfinite;
        ++check.failed;
        continue;
      }
      const double error = std::abs(actual - reference);
      const double tolerance = double(c.atol) + double(c.rtol) * std::abs(reference);
      check.max_error = std::max(check.max_error, error);
      check.max_tolerance = std::max(check.max_tolerance, tolerance);
      const double ratio = tolerance == 0.0 ? (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity()) : error / tolerance;
      check.max_ratio = std::max(check.max_ratio, ratio);
      if (error > tolerance) ++check.failed;
    }
  }
  for (std::size_t i = 0; i < y_count; ++i) {
    bool logical = false;
    if (c.rows > 0 && i >= std::size_t(c.y_offset)) {
      const std::size_t relative = i - std::size_t(c.y_offset);
      const std::size_t row = relative / c.y_stride;
      const std::size_t col = relative % c.y_stride;
      logical = row < std::size_t(c.rows) && col < std::size_t(c.cols);
    }
    if (!logical) {
      ++check.padding_checked;
      if (std::memcmp(&output[i], &kPaddingSentinel, sizeof(float)) != 0)
        ++check.padding_writes;
    }
  }
  return check;
}

struct TimedResult {
  std::vector<double> samples;
  std::string implementation_path;
};

TimedResult time_case(Case& c, const std::string& implementation,
                      cudaStream_t stream, int warmups, int repetitions,
                      int rotations) {
  if (c.rows == 0) throw std::invalid_argument("timing an empty zero-row case is not meaningful");
  std::vector<std::unique_ptr<DeviceFloats>> dx, dg, dy;
  std::vector<std::unique_ptr<DeviceBytes>> dm;
  const std::size_t y_count = output_extent(c);
  std::vector<float> initial_output(std::max<std::size_t>(y_count, 1), kPaddingSentinel);
  for (int i = 0; i < rotations; ++i) {
    dx.emplace_back(new DeviceFloats(c.x.size()));
    dg.emplace_back(new DeviceFloats(c.gamma.size()));
    dm.emplace_back(new DeviceBytes(c.mask.size()));
    dy.emplace_back(new DeviceFloats(y_count));
    if (!c.x.empty()) require_cuda(cudaMemcpyAsync(dx.back()->get(), c.x.data(), c.x.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "queue timing x");
    if (!c.gamma.empty()) require_cuda(cudaMemcpyAsync(dg.back()->get(), c.gamma.data(), c.gamma.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "queue timing gamma");
    if (!c.mask.empty()) require_cuda(cudaMemcpyAsync(dm.back()->get(), c.mask.data(), c.mask.size(), cudaMemcpyHostToDevice, stream), "queue timing mask");
    require_cuda(cudaMemcpyAsync(dy.back()->get(), initial_output.data(), initial_output.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "queue timing output initialization");
  }
  // Leave host-to-device setup outside every event interval, including the
  // one-shot profile mode with zero warmups.
  require_cuda(cudaStreamSynchronize(stream), "synchronize timing setup copies");
  TimedResult result;
  auto invoke = [&](int index) {
    kwv_rows::RmsnormCtaPath selected = kwv_rows::RmsnormCtaPath::kNoLaunch;
    launch(c, implementation, dx[index]->get(), dm[index]->get(), dg[index]->get(),
           dy[index]->get(), stream, &selected);
    result.implementation_path = implementation_path(c, implementation, selected);
  };
  for (int i = 0; i < warmups; ++i) invoke(i % rotations);
  require_cuda(cudaStreamSynchronize(stream), "synchronize timing warmups");
  cudaEvent_t start = nullptr, stop = nullptr;
  require_cuda(cudaEventCreate(&start), "create timing start event");
  require_cuda(cudaEventCreate(&stop), "create timing stop event");
  result.samples.reserve(repetitions);
  for (int i = 0; i < repetitions; ++i) {
    const int index = i % rotations;
    require_cuda(cudaEventRecord(start, stream), "record timing start");
    invoke(index);
    require_cuda(cudaEventRecord(stop, stream), "record timing stop");
    require_cuda(cudaEventSynchronize(stop), "wait timing stop");
    float ms = 0.0f;
    require_cuda(cudaEventElapsedTime(&ms, start, stop), "read timing sample");
    result.samples.push_back(double(ms) * 1000.0);
  }
  require_cuda(cudaEventDestroy(stop), "destroy timing stop event");
  require_cuda(cudaEventDestroy(start), "destroy timing start event");
  return result;
}

std::string json_escape(const std::string& in) {
  std::string out;
  for (char ch : in) {
    if (ch == '"' || ch == '\\') { out.push_back('\\'); out.push_back(ch); }
    else if (ch == '\n') out += "\\n";
    else if (ch == '\r') out += "\\r";
    else out.push_back(ch);
  }
  return out;
}

std::string uuid_hex(const cudaUUID_t& uuid) {
  static const char hex[] = "0123456789abcdef";
  std::string out;
  out.reserve(32);
  for (unsigned char byte : uuid.bytes) {
    out.push_back(hex[byte >> 4]);
    out.push_back(hex[byte & 15]);
  }
  return out;
}

void print_case(const Case& c, const CaseCheck& check, bool first) {
  if (!first) std::cout << ',';
  std::cout << std::setprecision(17)
      << "{\"case_id\":\"" << json_escape(c.id) << "\",\"operation\":\""
      << (c.softmax ? "softmax" : "rmsnorm") << "\",\"rows\":" << c.rows
      << ",\"cols\":" << c.cols << ",\"expected_outputs\":"
      << (std::uint64_t(c.rows) * c.cols) << ",\"implementation_path\":\""
      << check.selected_path << "\",\"checked_outputs\":" << check.checked
      << ",\"check_identity\":\"logical-row-major:" << json_escape(c.id) << "\""
      << ",\"failed_outputs\":" << check.failed
      << ",\"nonfinite_outputs\":" << check.nonfinite
      << ",\"padding_checked\":" << check.padding_checked
      << ",\"padding_writes\":" << check.padding_writes
      << ",\"failed_checks\":" << (check.failed + check.padding_writes)
      << ",\"max_abs_error\":" << check.max_error
      << ",\"max_admission_tolerance\":" << check.max_tolerance
      << ",\"max_tolerance_ratio\":" << check.max_ratio
      << ",\"atol\":" << c.atol << ",\"rtol\":" << c.rtol
      << ",\"passed\":" << ((check.checked == std::uint64_t(c.rows) * c.cols &&
                                    check.failed == 0 && check.nonfinite == 0 &&
                                    check.padding_writes == 0) ? "true" : "false") << '}';
}

int run(const Options& o) {
  auto cases = read_cases(o.input);
  read_references(o.reference, cases);
  std::vector<Case*> selected_cases;
  std::map<std::string, bool> selected_ids;
  if (o.case_ids.empty()) {
    for (auto& c : cases) selected_cases.push_back(&c);
  } else {
    for (const auto& id : o.case_ids) {
      if (selected_ids.emplace(id, true).second == false)
        throw std::invalid_argument("case ID is repeated: " + id);
      auto it = std::find_if(cases.begin(), cases.end(), [&](const Case& c) { return c.id == id; });
      if (it == cases.end()) throw std::invalid_argument("case is absent from bundle: " + id);
      selected_cases.push_back(&*it);
    }
  }
  for (Case* selected : selected_cases) {
    const Case& c = *selected;
    const auto route = effective_implementation(c, o.implementation);
    if (o.mode == "time" && c.rows == 0)
      throw std::invalid_argument("zero-row case cannot be timed: " + c.id);
    if (o.mode == "time" && !c.softmax && route == "vec4" && c.cols < 4)
      throw std::invalid_argument("selected vector implementation requires N>=4: " + c.id);
    if (c.softmax && route == "vec4")
      throw std::invalid_argument("vec4 is RMSNorm-only (use candidate for mixed operator routes)");
  }
  int device = 0;
  require_cuda(cudaGetDevice(&device), "query active CUDA device");
  cudaDeviceProp properties{};
  require_cuda(cudaGetDeviceProperties(&properties, device), "query active CUDA device properties");
  if (properties.major != 7 || properties.minor != 0)
    throw std::runtime_error("active CUDA device is not compute capability 7.0");
  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create nonblocking stream");
  const std::string uuid = uuid_hex(properties.uuid);

  if (o.mode == "time") {
    struct TimedCase { Case* input; TimedResult result; };
    std::vector<TimedCase> timed;
    timed.reserve(selected_cases.size());
    for (Case* selected : selected_cases) {
      Case& c = *selected;
      timed.push_back({&c, time_case(c, o.implementation, stream, o.warmups,
                                     o.repetitions, o.rotations)});
    }
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    std::cout << std::setprecision(17)
      << "{\"format\":\"KERNELWIKI-ROW-TIMING/1\",\"status\":\"completed\",\"gpu\":\""
      << json_escape(properties.name) << "\",\"gpu_uuid\":\"" << uuid
      << "\",\"compute_capability\":\"7.0\",\"implementation\":\"" << o.implementation
      << "\",\"warmups\":" << o.warmups << ",\"repetitions\":" << o.repetitions
      << ",\"rotation_count\":" << o.rotations << ",\"cases\":[";
    for (std::size_t i = 0; i < timed.size(); ++i) {
      const Case& c = *timed[i].input;
      const auto& samples = timed[i].result.samples;
      const std::uint64_t input_bytes = std::uint64_t(c.x.size()) * sizeof(float) +
          c.mask.size() + std::uint64_t(c.gamma.size()) * sizeof(float);
      const std::uint64_t output_bytes = output_extent(c) * sizeof(float);
      const std::uint64_t working_set = input_bytes + output_bytes;
      if (i) std::cout << ',';
      std::cout << "{\"case_id\":\"" << json_escape(c.id)
      << "\",\"operation\":\"" << (c.softmax ? "softmax" : "rmsnorm")
      << "\",\"implementation_path\":\"" << timed[i].result.implementation_path
      << "\",\"rows\":" << c.rows << ",\"cols\":" << c.cols
      << ",\"l2_bytes\":" << properties.l2CacheSize
      << ",\"input_bytes_per_buffer\":" << input_bytes
      << ",\"output_bytes_per_buffer\":" << output_bytes
      << ",\"workspace_bytes_per_buffer\":0"
      << ",\"working_set_bytes_per_buffer\":" << working_set
      << ",\"working_set_bytes_rotated\":" << working_set * o.rotations
      << ",\"cache_capacity_class\":\""
      << ((working_set * o.rotations > std::uint64_t(properties.l2CacheSize)) ? "exceeds_l2_capacity" : "within_l2_capacity")
      << "\",\"samples_us\":[";
      for (std::size_t j = 0; j < samples.size(); ++j) { if (j) std::cout << ','; std::cout << samples[j]; }
      std::cout << "]}";
    }
    std::cout << "]}\n";
    return 0;
  }

  bool all_passed = true;
  std::uint64_t total_expected = 0, total_checked = 0, total_failed = 0;
  std::uint64_t total_nonfinite = 0, total_padding_writes = 0;
  std::vector<CaseCheck> checks;
  checks.reserve(selected_cases.size());
  for (Case* selected : selected_cases) {
    Case& c = *selected;
    CaseCheck check = execute_correctness(c, o.implementation, stream);
    const std::uint64_t case_expected = std::uint64_t(c.rows) * c.cols;
    all_passed &= check.checked == case_expected && check.failed == 0 &&
                  check.nonfinite == 0 && check.padding_writes == 0;
    total_expected += case_expected;
    total_checked += check.checked;
    total_failed += check.failed;
    total_nonfinite += check.nonfinite;
    total_padding_writes += check.padding_writes;
    checks.push_back(check);
  }
  require_cuda(cudaStreamDestroy(stream), "destroy stream");
  std::cout << "{\"format\":\"KERNELWIKI-ROW-CHECK/1\",\"status\":\""
      << (all_passed && total_checked == total_expected ? "passed" : "failed") << "\",\"gpu\":\""
      << json_escape(properties.name) << "\",\"gpu_uuid\":\"" << uuid
      << "\",\"compute_capability\":\"7.0\",\"operator\":\"row_suite\",\"implementation\":\""
      << o.implementation << "\",\"case_scope\":\""
      << (o.case_ids.empty() ? "full_suite" : "selected_subset")
      << "\",\"case_count\":" << selected_cases.size()
      << ",\"expected_outputs\":" << total_expected
      << ",\"checked_outputs\":" << total_checked
      << ",\"failed_outputs\":" << total_failed
      << ",\"nonfinite_outputs\":" << total_nonfinite
      << ",\"padding_writes\":" << total_padding_writes << ",\"cases\":[";
  for (std::size_t i = 0; i < selected_cases.size(); ++i)
    print_case(*selected_cases[i], checks[i], i == 0);
  std::cout << "]}\n";
  return (all_passed && total_checked == total_expected) ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "row_runner: " << error.what() << '\n';
    std::cout << "{\"format\":\"KERNELWIKI-ROW-CHECK/1\",\"status\":\"failed\","
                 "\"error\":\"" << json_escape(error.what()) << "\"}\n";
    return 2;
  }
}
