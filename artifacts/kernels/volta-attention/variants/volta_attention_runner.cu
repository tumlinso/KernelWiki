#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "volta_attention_online.cuh"

namespace {
constexpr uint32_t kBundleVersion = 1;
constexpr unsigned kCausal = 1u;
constexpr unsigned kSelected = 2u;
constexpr unsigned kHasMask = 4u;
constexpr float kOutputAtol = 1.5e-4f;
constexpr float kOutputRtol = 2.0e-4f;
constexpr float kRowMaxAtol = 2.0e-5f;
constexpr float kNormalizerAtol = 1.0e-7f;
constexpr float kNormalizerRtol = 5.0e-4f;
constexpr int kTimingWarmups = 5;
constexpr int kTimingSamples = 7;
constexpr int kTimingRepeats = 8;

#pragma pack(push, 1)
struct BundleHeader { char magic[8]; uint32_t version; uint32_t case_count; };
struct InputRecord {
  char name[64];
  uint32_t b, hq, hkv, sq, sk, d, flags, block_size, max_selected, reserved;
  uint64_t q_offset, q_count, k_offset, k_count, v_offset, v_count;
  uint64_t mask_offset, mask_count, ids_offset, ids_count;
  uint64_t counts_offset, counts_count;
};
struct ReferenceRecord {
  char name[64];
  uint32_t b, hq, hkv, sq, sk, d, reserved1, reserved2;
  uint64_t output_offset, output_count, rows_offset, row_count;
};
struct ReferenceRow {
  double max_score;
  double normalizer;
  double logsumexp;
  uint64_t valid_keys;
};
#pragma pack(pop)
static_assert(sizeof(BundleHeader) == 16, "bundle header layout changed");
static_assert(sizeof(InputRecord) == 200, "input record layout changed");
static_assert(sizeof(ReferenceRecord) == 128, "reference record layout changed");
static_assert(sizeof(ReferenceRow) == 32, "reference row layout changed");

void cuda_check(cudaError_t status, const char* expression, int line) {
  if (status != cudaSuccess) {
    std::ostringstream out;
    out << "CUDA error at line " << line << " for " << expression << ": "
        << cudaGetErrorString(status);
    throw std::runtime_error(out.str());
  }
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr, __LINE__)

void cublas_check(cublasStatus_t status, const char* expression, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream out;
    out << "cuBLAS error at line " << line << " for " << expression
        << ": status=" << static_cast<int>(status);
    throw std::runtime_error(out.str());
  }
}
#define CUBLAS_CHECK(expr) cublas_check((expr), #expr, __LINE__)

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open bundle: " + path);
  input.seekg(0, std::ios::end);
  const auto end = input.tellg();
  if (end < 0) throw std::runtime_error("cannot size bundle: " + path);
  std::vector<uint8_t> bytes(static_cast<size_t>(end));
  input.seekg(0, std::ios::beg);
  if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) throw std::runtime_error("short read from bundle: " + path);
  return bytes;
}

template <typename T>
T load_record(const std::vector<uint8_t>& bytes, size_t offset, const char* label) {
  if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
    throw std::runtime_error(std::string(label) + " record is outside bundle bounds");
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  return value;
}

std::string record_name(const char raw[64]) {
  const void* end = std::memchr(raw, '\0', 64);
  const size_t n = end == nullptr ? 64 : static_cast<const char*>(end) - raw;
  if (n == 0) throw std::runtime_error("case ID is empty");
  return std::string(raw, n);
}

size_t checked_size(uint64_t count, size_t width, const char* label) {
  if (count > std::numeric_limits<size_t>::max() / width)
    throw std::runtime_error(std::string(label) + " byte count overflows size_t");
  return static_cast<size_t>(count) * width;
}

void validate_section(uint64_t offset, uint64_t count, size_t width,
                      size_t table_end, size_t file_size, const char* label) {
  if (count == 0) {
    if (offset != 0) throw std::runtime_error(std::string(label) + " empty section has offset");
    return;
  }
  const size_t bytes = checked_size(count, width, label);
  if (offset < table_end || offset > file_size || bytes > file_size - offset)
    throw std::runtime_error(std::string(label) + " section is outside bundle bounds");
}

struct InputCase {
  std::string id;
  int b = 0, hq = 0, hkv = 0, sq = 0, sk = 0, d = 0;
  bool causal = false;
  bool selected = false;
  bool has_mask = false;
  int block_size = 0, max_selected = 0;
  std::vector<half> q, k, v;
  std::vector<uint8_t> mask;
  std::vector<int32_t> selected_ids, selected_counts;
  size_t rows() const { return static_cast<size_t>(b) * hq * sq; }
  size_t qk_elements() const { return rows() * sk; }
  size_t output_elements() const { return rows() * d; }
  int kv_head(int query_head) const { return query_head / (hq / hkv); }
};

std::vector<InputCase> read_inputs(const std::string& path) {
  const auto bytes = read_file(path);
  const auto header = load_record<BundleHeader>(bytes, 0, "input header");
  if (std::memcmp(header.magic, "KWVATTN1", 8) != 0 ||
      header.version != kBundleVersion || header.case_count == 0)
    throw std::runtime_error("unsupported represented-input bundle header");
  const size_t table_end = sizeof(BundleHeader) +
                           static_cast<size_t>(header.case_count) * sizeof(InputRecord);
  if (table_end > bytes.size()) throw std::runtime_error("input record table is truncated");
  std::vector<InputCase> cases;
  cases.reserve(header.case_count);
  for (uint32_t i = 0; i < header.case_count; ++i) {
    const auto rec = load_record<InputRecord>(bytes, sizeof(BundleHeader) +
                                              i * sizeof(InputRecord), "input");
    InputCase c;
    c.id = record_name(rec.name);
    c.b = rec.b; c.hq = rec.hq; c.hkv = rec.hkv;
    c.sq = rec.sq; c.sk = rec.sk; c.d = rec.d;
    c.causal = (rec.flags & kCausal) != 0;
    c.selected = (rec.flags & kSelected) != 0;
    c.has_mask = (rec.flags & kHasMask) != 0;
    c.block_size = static_cast<int>(rec.block_size);
    c.max_selected = static_cast<int>(rec.max_selected);
    if ((rec.flags & ~(kCausal | kSelected | kHasMask)) != 0 || rec.reserved != 0 ||
        c.b <= 0 || c.hq <= 0 || c.hkv <= 0 || c.sq <= 0 || c.sk <= 0 ||
        c.d <= 0 || c.d > kernelwiki_volta_attention::kMaxHeadDim || c.hq % c.hkv != 0 ||
        (c.causal && c.sq > c.sk))
      throw std::runtime_error(c.id + ": invalid input contract or unsupported head dimension");
    const uint64_t rows = static_cast<uint64_t>(c.b) * c.hq * c.sq;
    const uint64_t q_expected = rows * c.d;
    const uint64_t kv_expected = static_cast<uint64_t>(c.b) * c.hkv * c.sk * c.d;
    if (rec.q_count != q_expected || rec.k_count != kv_expected || rec.v_count != kv_expected)
      throw std::runtime_error(c.id + ": Q/K/V counts disagree with shapes");
    validate_section(rec.q_offset, rec.q_count, sizeof(half), table_end, bytes.size(), "Q");
    validate_section(rec.k_offset, rec.k_count, sizeof(half), table_end, bytes.size(), "K");
    validate_section(rec.v_offset, rec.v_count, sizeof(half), table_end, bytes.size(), "V");
    validate_section(rec.mask_offset, rec.mask_count, sizeof(uint8_t), table_end, bytes.size(), "mask");
    c.q.resize(rec.q_count); c.k.resize(rec.k_count); c.v.resize(rec.v_count);
    std::memcpy(c.q.data(), bytes.data() + rec.q_offset, checked_size(rec.q_count, sizeof(half), "Q"));
    std::memcpy(c.k.data(), bytes.data() + rec.k_offset, checked_size(rec.k_count, sizeof(half), "K"));
    std::memcpy(c.v.data(), bytes.data() + rec.v_offset, checked_size(rec.v_count, sizeof(half), "V"));
    for (half x : c.q) if (!std::isfinite(__half2float(x))) throw std::runtime_error(c.id + ": non-finite Q");
    for (half x : c.k) if (!std::isfinite(__half2float(x))) throw std::runtime_error(c.id + ": non-finite K");
    for (half x : c.v) if (!std::isfinite(__half2float(x))) throw std::runtime_error(c.id + ": non-finite V");
    if (c.has_mask != (rec.mask_count != 0) ||
        (rec.mask_count != 0 && rec.mask_count != rows * c.sk))
      throw std::runtime_error(c.id + ": mask count disagrees with flags or shapes");
    if (c.has_mask) {
      c.mask.resize(rec.mask_count);
      std::memcpy(c.mask.data(), bytes.data() + rec.mask_offset, rec.mask_count);
      for (uint8_t value : c.mask) if (value > 1) throw std::runtime_error(c.id + ": mask byte is not bool");
    }
    if (c.selected) {
      if (c.block_size <= 0 || c.max_selected <= 0 ||
          rec.ids_count != rows * c.max_selected || rec.counts_count != rows)
        throw std::runtime_error(c.id + ": selected-block counts disagree with shapes");
      validate_section(rec.ids_offset, rec.ids_count, sizeof(int32_t), table_end, bytes.size(), "selected IDs");
      validate_section(rec.counts_offset, rec.counts_count, sizeof(int32_t), table_end, bytes.size(), "selected counts");
      c.selected_ids.resize(rec.ids_count); c.selected_counts.resize(rec.counts_count);
      std::memcpy(c.selected_ids.data(), bytes.data() + rec.ids_offset,
                  checked_size(rec.ids_count, sizeof(int32_t), "selected IDs"));
      std::memcpy(c.selected_counts.data(), bytes.data() + rec.counts_offset,
                  checked_size(rec.counts_count, sizeof(int32_t), "selected counts"));
      const int block_count = (c.sk + c.block_size - 1) / c.block_size;
      for (size_t row = 0; row < rows; ++row) {
        const int count = c.selected_counts[row];
        if (count < 0 || count > c.max_selected)
          throw std::runtime_error(c.id + ": selected count is outside capacity");
        for (int slot = 0; slot < count; ++slot) {
          const int32_t id = c.selected_ids[row * c.max_selected + slot];
          if (id < 0 || id >= block_count)
            throw std::runtime_error(c.id + ": selected block ID is outside sequence");
          for (int prior = 0; prior < slot; ++prior)
            if (id == c.selected_ids[row * c.max_selected + prior])
              throw std::runtime_error(c.id + ": duplicate selected block ID");
        }
      }
    } else if (rec.ids_offset || rec.ids_count || rec.counts_offset ||
               rec.counts_count || rec.max_selected) {
      throw std::runtime_error(c.id + ": unselected case has selection metadata");
    }
    cases.push_back(std::move(c));
  }
  return cases;
}

struct ExpectedRow { double max_score, normalizer, logsumexp; uint64_t valid_keys; };
struct ExpectedCase {
  std::string id;
  int b, hq, hkv, sq, sk, d;
  std::vector<double> output;
  std::vector<ExpectedRow> rows;
};

std::vector<ExpectedCase> read_references(const std::string& path,
                                          const std::vector<InputCase>& inputs) {
  const auto bytes = read_file(path);
  const auto header = load_record<BundleHeader>(bytes, 0, "reference header");
  if (std::memcmp(header.magic, "KWVREF01", 8) != 0 ||
      header.version != kBundleVersion || header.case_count != inputs.size())
    throw std::runtime_error("reference bundle header does not match input suite");
  const size_t table_end = sizeof(BundleHeader) +
                           static_cast<size_t>(header.case_count) * sizeof(ReferenceRecord);
  if (table_end > bytes.size()) throw std::runtime_error("reference record table is truncated");
  std::vector<ExpectedCase> result;
  result.reserve(header.case_count);
  for (uint32_t i = 0; i < header.case_count; ++i) {
    const auto rec = load_record<ReferenceRecord>(bytes, sizeof(BundleHeader) +
                                                   i * sizeof(ReferenceRecord), "reference");
    const auto& input = inputs[i];
    ExpectedCase e;
    e.id = record_name(rec.name);
    e.b = rec.b; e.hq = rec.hq; e.hkv = rec.hkv;
    e.sq = rec.sq; e.sk = rec.sk; e.d = rec.d;
    if (e.id != input.id || e.b != input.b || e.hq != input.hq ||
        e.hkv != input.hkv || e.sq != input.sq || e.sk != input.sk ||
        e.d != input.d || rec.reserved1 != 0 || rec.reserved2 != 0 ||
        rec.output_count != input.output_elements() || rec.row_count != input.rows())
      throw std::runtime_error(e.id + ": reference metadata differs from input case");
    validate_section(rec.output_offset, rec.output_count, sizeof(double), table_end,
                     bytes.size(), "reference output");
    validate_section(rec.rows_offset, rec.row_count, sizeof(ExpectedRow), table_end,
                     bytes.size(), "reference rows");
    e.output.resize(rec.output_count); e.rows.resize(rec.row_count);
    std::memcpy(e.output.data(), bytes.data() + rec.output_offset,
                checked_size(rec.output_count, sizeof(double), "reference output"));
    std::memcpy(e.rows.data(), bytes.data() + rec.rows_offset,
                checked_size(rec.row_count, sizeof(ExpectedRow), "reference rows"));
    result.push_back(std::move(e));
  }
  return result;
}

std::vector<InputCase> select_cases(const std::vector<InputCase>& all,
                                    const std::string& requested) {
  if (requested.empty() || requested == "all") return all;
  for (const auto& item : all) if (item.id == requested) return {item};
  throw std::runtime_error("unknown case ID: " + requested);
}

struct DeviceInfo { int device = -1; std::string uuid; std::string name; };
DeviceInfo inspect_device() {
  DeviceInfo result;
  CUDA_CHECK(cudaGetDevice(&result.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, result.device));
  if (prop.major != 7 || prop.minor != 0)
    throw std::runtime_error("expected SM70; observed compute capability " +
                             std::to_string(prop.major) + "." + std::to_string(prop.minor));
  result.name = prop.name;
  std::ostringstream uuid;
  uuid << std::hex << std::setfill('0');
  for (int i = 0; i < 16; ++i) uuid << std::setw(2)
      << (static_cast<unsigned>(prop.uuid.bytes[i]) & 0xffu);
  // The controller binds the canonical UUID from its lease; this native field
  // intentionally reports CUDA's raw device-property UUID bytes.
  result.uuid = uuid.str();
  return result;
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count = 0) : count_(count) {
    if (count_) CUDA_CHECK(cudaMalloc(&ptr_, checked_size(count_, sizeof(T), "device allocation")));
  }
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return ptr_; }
  size_t size() const { return count_; }
  void copy_from(const T* source, size_t count, cudaStream_t stream) {
    if (count != count_) throw std::runtime_error("device copy size mismatch");
    if (count_) CUDA_CHECK(cudaMemcpyAsync(ptr_, source, count * sizeof(T),
                                           cudaMemcpyHostToDevice, stream));
  }
  void copy_to(T* destination, size_t count, cudaStream_t stream) const {
    if (count != count_) throw std::runtime_error("device copy size mismatch");
    if (count_) CUDA_CHECK(cudaMemcpyAsync(destination, ptr_, count * sizeof(T),
                                           cudaMemcpyDeviceToHost, stream));
  }
 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

class NonblockingStream {
 public:
  NonblockingStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~NonblockingStream() {
    if (stream_) {
      // CaseContext synchronizes before releasing its buffers. Keep this
      // fallback for construction failures or future early-return paths.
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }
  NonblockingStream(const NonblockingStream&) = delete;
  NonblockingStream& operator=(const NonblockingStream&) = delete;
  cudaStream_t get() const { return stream_; }
  void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
 private:
  cudaStream_t stream_ = nullptr;
};

__device__ bool key_is_valid(int row, int query_row, int key, int sq, int sk,
                             bool causal, int block_size, int max_selected,
                             const uint8_t* mask, const int32_t* selected_ids,
                             const int32_t* selected_counts) {
  if (causal && key > sk - sq + query_row) return false;
  if (mask != nullptr && mask[static_cast<int64_t>(row) * sk + key] != 0) return false;
  if (selected_counts != nullptr) {
    const int count = selected_counts[row];
    const int block = key / block_size;
    bool found = false;
    for (int slot = 0; slot < count; ++slot) {
      if (selected_ids[row * max_selected + slot] == block) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

__global__ void half_to_float(const half* source, float* destination, size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) destination[index] = __half2float(source[index]);
}

__global__ void materialized_softmax(const float* scores, float* probabilities,
                                     int rows, int hq, int sq, int sk, int d,
                                     bool causal, int block_size, int max_selected,
                                     const uint8_t* mask,
                                     const int32_t* selected_ids,
                                     const int32_t* selected_counts,
                                     float* row_max, float* row_normalizer) {
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (row >= rows || blockDim.x != 128) return;
  const int query_row = row % sq;
  const float scale = rsqrtf(static_cast<float>(d));
  __shared__ float reduction[128];
  float local_max = -INFINITY;
  for (int key = tid; key < sk; key += blockDim.x) {
    if (key_is_valid(row, query_row, key, sq, sk, causal,
                     block_size, max_selected, mask,
                     selected_ids, selected_counts)) {
      local_max = fmaxf(local_max,
                        scores[static_cast<int64_t>(row) * sk + key] * scale);
    }
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    __syncthreads();
  }
  const float maximum = reduction[0];
  float local_sum = 0.0f;
  if (isfinite(maximum)) {
    for (int key = tid; key < sk; key += blockDim.x) {
      if (key_is_valid(row, query_row, key, sq, sk, causal,
                       block_size, max_selected, mask,
                       selected_ids, selected_counts)) {
        local_sum += expf(scores[static_cast<int64_t>(row) * sk + key] * scale - maximum);
      }
    }
  }
  reduction[tid] = local_sum;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }
  const float normalizer = reduction[0];
  if (tid == 0) {
    row_max[row] = normalizer == 0.0f ? -INFINITY : maximum;
    row_normalizer[row] = normalizer;
  }
  for (int key = tid; key < sk; key += blockDim.x) {
    float probability = 0.0f;
    if (normalizer > 0.0f && key_is_valid(row, query_row, key, sq, sk, causal,
                                          block_size, max_selected, mask,
                                          selected_ids, selected_counts)) {
      probability = expf(scores[static_cast<int64_t>(row) * sk + key] * scale - maximum) /
                    normalizer;
    }
    probabilities[static_cast<int64_t>(row) * sk + key] = probability;
  }
}

struct DeviceResult {
  std::vector<float> output;
  std::vector<float> row_max;
  std::vector<float> row_normalizer;
};

class CaseContext {
 public:
  explicit CaseContext(const InputCase& c, int splits)
      : host_(c), split_count_(splits), rows_(c.rows()),
        d_q_(c.q.size()), d_k_(c.k.size()), d_v_(c.v.size()),
        d_output_(c.output_elements()), d_row_max_(rows_),
        d_row_normalizer_(rows_), d_mask_(c.mask.size()),
        d_selected_ids_(c.selected_ids.size()), d_selected_counts_(c.selected_counts.size()),
        d_scores_(c.qk_elements()), d_probabilities_(c.qk_elements()),
        d_v_float_(c.v.size()),
        d_partial_m_(splits > 1 ? rows_ * splits : 0),
        d_partial_l_(splits > 1 ? rows_ * splits : 0),
        d_partial_u_(splits > 1 ? rows_ * splits * c.d : 0) {
    if (splits <= 0) throw std::runtime_error("split count must be positive");
    try {
      // Queue every upload on the same explicit stream as the compute path.
      // Synchronize before returning so pageable InputCase storage may safely
      // be released by the caller after this context's lifetime.
      d_q_.copy_from(c.q.data(), c.q.size(), stream_.get());
      d_k_.copy_from(c.k.data(), c.k.size(), stream_.get());
      d_v_.copy_from(c.v.data(), c.v.size(), stream_.get());
      if (!c.mask.empty()) d_mask_.copy_from(c.mask.data(), c.mask.size(), stream_.get());
      if (!c.selected_ids.empty())
        d_selected_ids_.copy_from(c.selected_ids.data(), c.selected_ids.size(), stream_.get());
      if (!c.selected_counts.empty())
        d_selected_counts_.copy_from(c.selected_counts.data(), c.selected_counts.size(), stream_.get());
      CUBLAS_CHECK(cublasCreate(&handle_));
      CUBLAS_CHECK(cublasSetStream(handle_, stream_.get()));
      stream_.synchronize();
    } catch (...) {
      // A partially queued pageable H2D upload can still reference its source
      // staging data. Finish that stream before the failed constructor releases
      // its device members or returns control to the caller.
      if (stream_.get()) cudaStreamSynchronize(stream_.get());
      if (handle_) {
        cublasDestroy(handle_);
        handle_ = nullptr;
      }
      throw;
    }
  }
  ~CaseContext() {
    // Complete queued work before member DeviceBuffers and the host InputCase
    // reference can leave scope. All expected paths already synchronize, but
    // this makes early exits safe as well.
    if (stream_.get()) cudaStreamSynchronize(stream_.get());
    if (handle_) cublasDestroy(handle_);
  }
  CaseContext(const CaseContext&) = delete;
  CaseContext& operator=(const CaseContext&) = delete;

  const InputCase& host() const { return host_; }
  cublasHandle_t handle() const { return handle_; }
  cudaStream_t stream() const { return stream_.get(); }
  void synchronize() const { stream_.synchronize(); }

  void launch_baseline() {
    const size_t values = host_.v.size();
    half_to_float<<<static_cast<unsigned>((values + 255) / 256), 256, 0, stream()>>>(
        d_v_.get(), d_v_float_.get(), values);
    CUDA_CHECK(cudaGetLastError());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int batch = 0; batch < host_.b; ++batch) {
      for (int head = 0; head < host_.hq; ++head) {
        const int kv_head = host_.kv_head(head);
        const int64_t qbase =
            (static_cast<int64_t>(batch) * host_.hq + head) * host_.sq * host_.d;
        const int64_t kbase =
            (static_cast<int64_t>(batch) * host_.hkv + kv_head) * host_.sk * host_.d;
        const int64_t row_base =
            (static_cast<int64_t>(batch) * host_.hq + head) * host_.sq;
        const int64_t score_base = row_base * host_.sk;
        CUBLAS_CHECK(cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N,
                                  host_.sk, host_.sq, host_.d,
                                  &alpha, d_k_.get() + kbase, CUDA_R_16F, host_.d,
                                  d_q_.get() + qbase, CUDA_R_16F, host_.d,
                                  &beta, d_scores_.get() + score_base,
                                  CUDA_R_32F, host_.sk,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
    }
    materialized_softmax<<<static_cast<unsigned>(rows_), 128, 0, stream()>>>(
        d_scores_.get(), d_probabilities_.get(), static_cast<int>(rows_),
        host_.hq, host_.sq, host_.sk, host_.d, host_.causal, host_.block_size,
        host_.max_selected, host_.mask.empty() ? nullptr : d_mask_.get(),
        host_.selected ? d_selected_ids_.get() : nullptr,
        host_.selected ? d_selected_counts_.get() : nullptr,
        d_row_max_.get(), d_row_normalizer_.get());
    CUDA_CHECK(cudaGetLastError());
    for (int batch = 0; batch < host_.b; ++batch) {
      for (int head = 0; head < host_.hq; ++head) {
        const int kv_head = host_.kv_head(head);
        const int64_t vbase =
            (static_cast<int64_t>(batch) * host_.hkv + kv_head) * host_.sk * host_.d;
        const int64_t row_base =
            (static_cast<int64_t>(batch) * host_.hq + head) * host_.sq;
        const int64_t probability_base = row_base * host_.sk;
        const int64_t output_base = row_base * host_.d;
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 host_.d, host_.sq, host_.sk,
                                 &alpha, d_v_float_.get() + vbase, host_.d,
                                 d_probabilities_.get() + probability_base, host_.sk,
                                 &beta, d_output_.get() + output_base, host_.d));
      }
    }
  }

  void launch_online(int splits) {
    if (splits != split_count_) throw std::runtime_error("context split allocation mismatch");
    const dim3 grid(static_cast<unsigned>(rows_), static_cast<unsigned>(splits), 1);
    kernelwiki_volta_attention::online_rows<<<grid, 128, 0, stream()>>>(
        d_q_.get(), d_k_.get(), d_v_.get(),
        host_.mask.empty() ? nullptr : d_mask_.get(),
        host_.selected ? d_selected_ids_.get() : nullptr,
        host_.selected ? d_selected_counts_.get() : nullptr,
        static_cast<int>(rows_), host_.hq, host_.hkv, host_.sq, host_.sk, host_.d,
        host_.causal, host_.block_size, host_.max_selected, splits,
        d_output_.get(), d_row_max_.get(), d_row_normalizer_.get(),
        splits > 1 ? d_partial_m_.get() : nullptr,
        splits > 1 ? d_partial_l_.get() : nullptr,
        splits > 1 ? d_partial_u_.get() : nullptr);
    CUDA_CHECK(cudaGetLastError());
    if (splits > 1) {
      kernelwiki_volta_attention::merge_split_kv<<<static_cast<unsigned>(rows_), 128, 0, stream()>>>(
          d_partial_m_.get(), d_partial_l_.get(), d_partial_u_.get(),
          static_cast<int>(rows_), host_.d, splits,
          d_output_.get(), d_row_max_.get(), d_row_normalizer_.get());
      CUDA_CHECK(cudaGetLastError());
    }
  }

  DeviceResult copy_result() const {
    DeviceResult result;
    result.output.resize(host_.output_elements());
    result.row_max.resize(rows_);
    result.row_normalizer.resize(rows_);
    d_output_.copy_to(result.output.data(), result.output.size(), stream());
    d_row_max_.copy_to(result.row_max.data(), result.row_max.size(), stream());
    d_row_normalizer_.copy_to(result.row_normalizer.data(), result.row_normalizer.size(), stream());
    // The host result vectors are pageable and are read immediately afterward.
    // Keep this verification-only transfer and its synchronization outside timing.
    synchronize();
    return result;
  }

 private:
  const InputCase& host_;
  int split_count_;
  size_t rows_;
  NonblockingStream stream_;
  DeviceBuffer<half> d_q_, d_k_, d_v_;
  DeviceBuffer<float> d_output_, d_row_max_, d_row_normalizer_;
  DeviceBuffer<uint8_t> d_mask_;
  DeviceBuffer<int32_t> d_selected_ids_, d_selected_counts_;
  DeviceBuffer<float> d_scores_, d_probabilities_, d_v_float_;
  DeviceBuffer<float> d_partial_m_, d_partial_l_, d_partial_u_;
  cublasHandle_t handle_ = nullptr;
};

void launch_implementation(CaseContext& context, const std::string& implementation,
                           int splits) {
  if (implementation == "baseline") context.launch_baseline();
  else if (implementation == "online") context.launch_online(1);
  else if (implementation == "split") context.launch_online(splits);
  else throw std::runtime_error("implementation must be baseline, online, or split");
}

struct CheckReport {
  bool passed = true;
  double max_abs_output_error = 0.0;
  double max_row_max_abs_error = 0.0;
  double max_row_normalizer_rel_error = 0.0;
  size_t checked_outputs = 0;
  size_t checked_rows = 0;
  std::string failure_component;
  size_t failure_index = 0;
  double expected = 0.0;
  double actual = 0.0;
  double allowed = 0.0;
};

CheckReport compare_result(const InputCase& input, const ExpectedCase& expected,
                           const DeviceResult& actual) {
  CheckReport report;
  if (actual.output.size() != expected.output.size() ||
      actual.row_max.size() != expected.rows.size() ||
      actual.row_normalizer.size() != expected.rows.size()) {
    report.passed = false;
    report.failure_component = "result_shape";
    return report;
  }
  for (size_t i = 0; i < expected.output.size(); ++i) {
    const double ref = expected.output[i];
    const double got = actual.output[i];
    const double error = std::abs(got - ref);
    const double allowed = kOutputAtol + kOutputRtol * std::abs(ref);
    ++report.checked_outputs;
    if (std::isfinite(error)) report.max_abs_output_error = std::max(report.max_abs_output_error, error);
    if (!std::isfinite(got) || !std::isfinite(ref) || error > allowed) {
      report.passed = false;
      report.failure_component = "output";
      report.failure_index = i;
      report.expected = ref; report.actual = got; report.allowed = allowed;
      return report;
    }
  }
  for (size_t i = 0; i < expected.rows.size(); ++i) {
    const auto& ref = expected.rows[i];
    const double got_m = actual.row_max[i];
    const double got_l = actual.row_normalizer[i];
    ++report.checked_rows;
    if (ref.valid_keys == 0) {
      const bool max_is_negative_inf = std::isinf(got_m) && std::signbit(got_m);
      if (!max_is_negative_inf || got_l != 0.0) {
        report.passed = false;
        report.failure_component = "empty_row_state";
        report.failure_index = i;
        report.expected = 0.0; report.actual = got_l; report.allowed = 0.0;
        return report;
      }
      const size_t row_base = i * static_cast<size_t>(input.d);
      for (int feature = 0; feature < input.d; ++feature) {
        const double value = actual.output[row_base + feature];
        if (value != 0.0) {
          report.passed = false;
          report.failure_component = "empty_row_output";
          report.failure_index = row_base + feature;
          report.expected = 0.0; report.actual = value; report.allowed = 0.0;
          return report;
        }
      }
      continue;
    }
    const double max_error = std::abs(got_m - ref.max_score);
    report.max_row_max_abs_error = std::max(report.max_row_max_abs_error, max_error);
    if (!std::isfinite(got_m) || max_error > kRowMaxAtol) {
      report.passed = false;
      report.failure_component = "row_max";
      report.failure_index = i;
      report.expected = ref.max_score; report.actual = got_m; report.allowed = kRowMaxAtol;
      return report;
    }
    const double norm_error = std::abs(got_l - ref.normalizer);
    const double norm_relative = norm_error / std::max(std::abs(ref.normalizer), 1.0e-30);
    report.max_row_normalizer_rel_error = std::max(report.max_row_normalizer_rel_error,
                                                    norm_relative);
    const double norm_allowed = kNormalizerAtol + kNormalizerRtol * std::abs(ref.normalizer);
    if (!std::isfinite(got_l) || got_l <= 0.0 || norm_error > norm_allowed) {
      report.passed = false;
      report.failure_component = "row_normalizer";
      report.failure_index = i;
      report.expected = ref.normalizer; report.actual = got_l; report.allowed = norm_allowed;
      return report;
    }
  }
  (void)input;
  return report;
}

struct Options {
  std::string case_id = "all";
  std::string input_path;
  std::string reference_path;
  std::string implementation = "online";
  std::string mode = "correctness";
  int splits = 1;
  int warmups = 2;
  int repetitions = 11;
};

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) out << "\\u00" << std::hex << std::setw(2)
                           << std::setfill('0') << static_cast<int>(c) << std::dec;
        else out << static_cast<char>(c);
    }
  }
  return out.str();
}

void json_number(std::ostream& out, double value) {
  if (std::isfinite(value)) out << value;
  else out << "null";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (++i >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
      return argv[i];
    };
    if (arg == "--case") options.case_id = value("--case");
    else if (arg == "--input") options.input_path = value("--input");
    else if (arg == "--reference") options.reference_path = value("--reference");
    else if (arg == "--impl") options.implementation = value("--impl");
    else if (arg == "--mode") options.mode = value("--mode");
    else if (arg == "--splits") options.splits = std::stoi(value("--splits"));
    else if (arg == "--warmups") options.warmups = std::stoi(value("--warmups"));
    else if (arg == "--repetitions") options.repetitions = std::stoi(value("--repetitions"));
    else if (arg == "--help") {
      std::cout << "usage: volta_attention_runner --input <KWVATTN1> --reference <KWVREF01> "
                   "[--case all|CASE_ID] [--impl baseline|online|split] "
                   "[--splits N] [--mode correctness|time|single] "
                   "[--warmups N] [--repetitions N]\\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (options.input_path.empty() || options.reference_path.empty())
    throw std::invalid_argument("--input and --reference are required");
  if (options.implementation != "baseline" && options.implementation != "online" &&
      options.implementation != "split")
    throw std::invalid_argument("--impl must be baseline, online, or split");
  if (options.mode != "correctness" && options.mode != "time" && options.mode != "single")
    throw std::invalid_argument("--mode must be correctness, time, or single");
  if (options.splits <= 0 || options.splits > 64 || options.warmups < 0 ||
      options.repetitions <= 0)
    throw std::invalid_argument("invalid split, warmup, or repetition count");
  if (options.implementation == "split" && options.splits < 2)
    throw std::invalid_argument("split implementation requires --splits >= 2");
  if (options.implementation != "split" && options.splits != 1)
    throw std::invalid_argument("--splits applies only to --impl split");
  if (options.mode == "single" && options.case_id == "all")
    throw std::invalid_argument("--mode single requires one explicit --case");
  return options;
}

struct CaseCheck {
  std::string id;
  CheckReport check;
  std::vector<double> samples_us;
};

struct RunCheck {
  std::vector<CaseCheck> cases;
  bool passed = true;
};

RunCheck verify_implementation(const std::vector<InputCase>& selected,
                               const std::vector<ExpectedCase>& references,
                               const std::string& implementation, int splits) {
  RunCheck run;
  for (const auto& input : selected) {
    auto reference = std::find_if(references.begin(), references.end(),
                                  [&](const ExpectedCase& item) {
                                    return item.id == input.id;
                                  });
    if (reference == references.end())
      throw std::runtime_error(input.id + ": no FP64 reference is loaded");
    const int allocated_splits = implementation == "split" ? splits : 1;
    CaseContext context(input, allocated_splits);
    launch_implementation(context, implementation, splits);
    const auto actual = context.copy_result();
    auto report = compare_result(input, *reference, actual);
    run.passed = run.passed && report.passed;
    run.cases.push_back({input.id, std::move(report), {}});
  }
  return run;
}

void write_run_json(std::ostream& out, const DeviceInfo& device,
                    const std::string& mode, const std::string& implementation,
                    const RunCheck& run, int splits) {
  out << std::setprecision(17)
      << "{\"format\":\"KERNELWIKI-ATTENTION-RUN/1\",\"status\":\""
      << (run.passed ? "passed" : "failed")
      << "\",\"compute_capability\":\"7.0\",\"gpu_uuid\":\""
      << json_escape(device.uuid) << "\",\"gpu\":\""
      << json_escape(device.name) << "\",\"case_id\":\"attention\",\"case\":\""
      << json_escape(run.cases.size() == 1 ? run.cases.front().id : "all")
      << "\",\"mode\":\"" << json_escape(mode)
      << "\",\"implementation\":\"" << json_escape(implementation)
      << "\",\"splits\":" << (implementation == "split" ? splits : 1)
      << ",\"checks\":[";
  bool first = true;
  const CaseCheck* first_failure = nullptr;
  for (const auto& item : run.cases) {
    if (!first) out << ',';
    first = false;
    const auto& check = item.check;
    out << "{\"case\":\"" << json_escape(item.id)
        << "\",\"passed\":" << (check.passed ? "true" : "false")
        << ",\"checked_outputs\":" << check.checked_outputs
        << ",\"checked_rows\":" << check.checked_rows
      << ",\"max_abs_output_error\":";
    json_number(out, check.max_abs_output_error);
    out << ",\"max_row_max_abs_error\":";
    json_number(out, check.max_row_max_abs_error);
    out << ",\"max_row_normalizer_relative_error\":";
    json_number(out, check.max_row_normalizer_rel_error);
    if (!check.passed) {
      out << ",\"failure_component\":\"" << json_escape(check.failure_component)
          << "\",\"failure_index\":" << check.failure_index
          << ",\"expected\":";
      json_number(out, check.expected);
      out << ",\"actual\":";
      json_number(out, check.actual);
      out << ",\"allowed\":";
      json_number(out, check.allowed);
      if (first_failure == nullptr) first_failure = &item;
    }
    if (mode == "time") {
      out << ",\"timing_phase\":\"steady_state_compute\",\"timing_scope\":{"
          << "\"included\":\"full selected implementation; baseline includes V expansion, QK, FP32 softmax, and PV; split includes partial and merge kernels\","
          << "\"excluded\":\"input loading, device allocation, and H2D upload\"},\"samples_us\":[";
      for (size_t i = 0; i < item.samples_us.size(); ++i) {
        if (i) out << ',';
        out << item.samples_us[i];
      }
      out << ']';
    }
    out << '}';
  }
  double max_output_error = 0.0;
  double max_row_max_error = 0.0;
  double max_normalizer_relative_error = 0.0;
  size_t checked_outputs = 0;
  size_t checked_rows = 0;
  for (const auto& item : run.cases) {
    max_output_error = std::max(max_output_error, item.check.max_abs_output_error);
    max_row_max_error = std::max(max_row_max_error, item.check.max_row_max_abs_error);
    max_normalizer_relative_error = std::max(max_normalizer_relative_error,
                                              item.check.max_row_normalizer_rel_error);
    checked_outputs += item.check.checked_outputs;
    checked_rows += item.check.checked_rows;
  }
  out << "],\"errors\":{\"checked_outputs\":" << checked_outputs
      << ",\"checked_rows\":" << checked_rows << ",\"max_abs_output_error\":";
  json_number(out, max_output_error);
  out << ",\"max_row_max_abs_error\":";
  json_number(out, max_row_max_error);
  out << ",\"max_row_normalizer_relative_error\":";
  json_number(out, max_normalizer_relative_error);
  out << "},\"failure\":";
  if (first_failure == nullptr) {
    out << "null";
  } else {
    out << "{\"case\":\"" << json_escape(first_failure->id)
        << "\",\"component\":\""
        << json_escape(first_failure->check.failure_component)
        << "\",\"index\":" << first_failure->check.failure_index
        << ",\"expected\":";
    json_number(out, first_failure->check.expected);
    out << ",\"actual\":";
    json_number(out, first_failure->check.actual);
    out << ",\"allowed\":";
    json_number(out, first_failure->check.allowed);
    out << '}';
  }
  out << "}\n";
}

int run(const Options& options) {
  auto all_inputs = read_inputs(options.input_path);
  auto all_references = read_references(options.reference_path, all_inputs);
  auto inputs = select_cases(all_inputs, options.case_id);
  const auto device = inspect_device();
  auto checked = verify_implementation(inputs, all_references,
                                        options.implementation, options.splits);
  if (!checked.passed) {
    write_run_json(std::cout, device, options.mode, options.implementation,
                   checked, options.splits);
    return 3;
  }
  if (options.mode == "correctness") {
    write_run_json(std::cout, device, options.mode, options.implementation,
                   checked, options.splits);
    return 0;
  }

  if (options.mode == "time") {
    for (size_t case_index = 0; case_index < inputs.size(); ++case_index) {
      const auto& input = inputs[case_index];
      auto& case_samples = checked.cases[case_index].samples_us;
      CaseContext context(input, options.implementation == "split" ? options.splits : 1);
      for (int i = 0; i < options.warmups; ++i)
        launch_implementation(context, options.implementation, options.splits);
      context.synchronize();
      cudaEvent_t start = nullptr, stop = nullptr;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      for (int i = 0; i < options.repetitions; ++i) {
        CUDA_CHECK(cudaEventRecord(start, context.stream()));
        launch_implementation(context, options.implementation, options.splits);
        CUDA_CHECK(cudaEventRecord(stop, context.stream()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        case_samples.push_back(static_cast<double>(elapsed_ms) * 1000.0);
      }
      CUDA_CHECK(cudaEventDestroy(stop));
      CUDA_CHECK(cudaEventDestroy(start));
    }
  } else {
    // `single` is intended for a selected profiler invocation. Correctness was
    // already checked above, and this is the one additional launch sequence.
    const auto& input = inputs.front();
    CaseContext context(input, options.implementation == "split" ? options.splits : 1);
    launch_implementation(context, options.implementation, options.splits);
    context.synchronize();
  }
  write_run_json(std::cout, device, options.mode, options.implementation,
                 checked, options.splits);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    return run(options);
  } catch (const std::exception& error) {
    std::cout << "{\"format\":\"KERNELWIKI-ATTENTION-RUN/1\","
              << "\"status\":\"failed\",\"case_id\":\"attention\","
              << "\"case\":\"unknown\",\"checks\":[],\"failure\":{"
              << "\"component\":\"runner\",\"message\":\""
              << json_escape(error.what()) << "\"},\"errors\":\"runner failed before a complete correctness record\"}\n";
    return 2;
  }
}
