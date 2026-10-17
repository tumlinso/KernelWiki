#pragma once

// Host-only represented-input and FP64-reference reader shared by the native
// runner and its CPU contract test. All numeric payloads are little-endian.
#include "sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace kwv08 {

constexpr double kAtol = 2.0e-5;
constexpr double kRtol = 2.0e-4;
constexpr float kOutputSentinel = 12345.25f;
constexpr std::size_t kBundleHeaderBytes = 8 + 4 + 4 + 64;

struct BundleCase {
  std::string id;
  std::uint32_t T=0, B=0, d_v=0, d_k=0;
  std::vector<float> keys, values, queries, alpha, beta;
  std::vector<std::int32_t> lengths;
  std::vector<float> initial_states;
  std::vector<double> expected_outputs, expected_final_states;
};

struct Bundle {
  std::string payload_sha256;
  std::string file_sha256;
  std::vector<BundleCase> cases;
};

inline std::uint32_t read_u32_le(const std::vector<std::uint8_t>& bytes,
                                 std::size_t& offset) {
  if (offset + 4 > bytes.size()) throw std::runtime_error("truncated uint32 in bundle");
  const std::uint32_t value = static_cast<std::uint32_t>(bytes[offset]) |
      (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
      (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
      (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
  offset += 4;
  return value;
}

inline std::size_t checked_product(std::initializer_list<std::uint32_t> dims) {
  std::size_t result = 1;
  for (const std::uint32_t dim : dims) {
    if (dim == 0 || result > std::numeric_limits<std::size_t>::max() / dim)
      throw std::runtime_error("zero/overflowing recurrence bundle extent");
    result *= dim;
  }
  return result;
}

template <typename T>
inline std::vector<T> read_array(const std::vector<std::uint8_t>& bytes,
                                 std::size_t& offset, std::size_t count) {
  static_assert(std::is_trivially_copyable<T>::value, "bundle array type must be trivial");
  if (count > (bytes.size() - std::min(offset, bytes.size())) / sizeof(T))
    throw std::runtime_error("truncated recurrence bundle array");
  std::vector<T> result(count);
  const std::size_t nbytes = count * sizeof(T);
  std::memcpy(result.data(), bytes.data() + offset, nbytes);
  offset += nbytes;
  return result;
}

inline void validate_case(const BundleCase& c, std::size_t b_dv);

inline Bundle read_bundle(const std::string& path) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "runner requires IEEE-754 binary32");
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                "runner requires IEEE-754 binary64");
  const std::uint16_t endian_probe = 1;
  if (*reinterpret_cast<const std::uint8_t*>(&endian_probe) != 1)
    throw std::runtime_error("bundle reader requires a little-endian host");

  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open represented recurrence bundle: " + path);
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < static_cast<std::streamoff>(kBundleHeaderBytes) ||
      size > static_cast<std::streamoff>(1024ull * 1024ull * 1024ull))
    throw std::runtime_error("recurrence bundle size is outside supported range");
  in.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  in.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!in) throw std::runtime_error("failed to read complete recurrence bundle");

  if (std::memcmp(bytes.data(), "KWV08BND", 8) != 0)
    throw std::runtime_error("bad recurrence bundle magic");
  std::size_t offset = 8;
  const std::uint32_t version = read_u32_le(bytes, offset);
  const std::uint32_t case_count = read_u32_le(bytes, offset);
  if (version != 1 || case_count == 0 || case_count > 64)
    throw std::runtime_error("unsupported recurrence bundle version/case count");
  if (offset + 64 > bytes.size()) throw std::runtime_error("truncated payload hash");
  std::string declared_payload_sha(reinterpret_cast<const char*>(bytes.data() + offset), 64);
  offset += 64;
  const std::string actual_payload_sha = sha256_hex(bytes.data() + offset, bytes.size() - offset);
  if (declared_payload_sha != actual_payload_sha)
    throw std::runtime_error("recurrence bundle payload SHA-256 mismatch");

  Bundle bundle;
  bundle.payload_sha256 = actual_payload_sha;
  bundle.file_sha256 = sha256_hex(bytes);
  std::set<std::string> case_ids;
  for (std::uint32_t case_index = 0; case_index < case_count; ++case_index) {
    const std::uint32_t name_bytes = read_u32_le(bytes, offset);
    if (name_bytes == 0 || name_bytes > 255 || offset + name_bytes > bytes.size())
      throw std::runtime_error("invalid recurrence case ID length");
    BundleCase c;
    c.id.assign(reinterpret_cast<const char*>(bytes.data() + offset), name_bytes);
    offset += name_bytes;
    if (!case_ids.insert(c.id).second) throw std::runtime_error("duplicate recurrence case ID");
    c.T = read_u32_le(bytes, offset);
    c.B = read_u32_le(bytes, offset);
    c.d_v = read_u32_le(bytes, offset);
    c.d_k = read_u32_le(bytes, offset);
    if (c.T > 4096 || c.B > 4096 || c.d_v > 256 || c.d_k > 256)
      throw std::runtime_error("recurrence case exceeds the bounded native runner");
    const auto t_b = checked_product({c.T, c.B});
    const auto t_b_dv = checked_product({c.T, c.B, c.d_v});
    const auto t_b_dk = checked_product({c.T, c.B, c.d_k});
    const auto b_dv_dk = checked_product({c.B, c.d_v, c.d_k});
    const auto state_per_sequence = checked_product({c.d_v, c.d_k});
    c.keys = read_array<float>(bytes, offset, t_b_dk);
    c.values = read_array<float>(bytes, offset, t_b_dv);
    c.queries = read_array<float>(bytes, offset, t_b_dk);
    c.alpha = read_array<float>(bytes, offset, t_b);
    c.beta = read_array<float>(bytes, offset, t_b);
    c.lengths = read_array<std::int32_t>(bytes, offset, c.B);
    c.initial_states = read_array<float>(bytes, offset, b_dv_dk);
    c.expected_outputs = read_array<double>(bytes, offset, t_b_dv);
    c.expected_final_states = read_array<double>(bytes, offset, b_dv_dk);
    validate_case(c, state_per_sequence);
    bundle.cases.push_back(std::move(c));
  }
  if (offset != bytes.size()) throw std::runtime_error("trailing bytes in recurrence bundle");
  return bundle;
}

inline void validate_case(const BundleCase& c, std::size_t state_per_sequence) {
  if (c.lengths.size() != c.B) throw std::runtime_error("wrong lengths extent");
  for (float x : c.initial_states)
    if (!std::isfinite(x)) throw std::runtime_error("non-finite initial recurrent state");
  for (std::uint32_t b = 0; b < c.B; ++b) {
    const std::int32_t length = c.lengths[b];
    if (length < 0 || static_cast<std::uint32_t>(length) > c.T)
      throw std::runtime_error("sequence length is outside [0,T]");
    for (std::uint32_t t = 0; t < c.T; ++t) {
      const bool live = t < static_cast<std::uint32_t>(length);
      const std::size_t gate = static_cast<std::size_t>(t) * c.B + b;
      if (live) {
        if (!std::isfinite(c.alpha[gate]) || !std::isfinite(c.beta[gate]) ||
            c.alpha[gate] < 0.0f || c.alpha[gate] > 1.0f ||
            c.beta[gate] < 0.0f || c.beta[gate] > 1.0f)
          throw std::runtime_error("live gates must be finite and in [0,1]");
        double norm2 = 0.0;
        for (std::uint32_t j = 0; j < c.d_k; ++j) {
          const std::size_t index = (static_cast<std::size_t>(t) * c.B + b) * c.d_k + j;
          if (!std::isfinite(c.keys[index]) || !std::isfinite(c.queries[index]))
            throw std::runtime_error("live key/query contains NaN or Inf");
          norm2 += static_cast<double>(c.keys[index]) * c.keys[index];
        }
        if (std::abs(std::sqrt(norm2) - 1.0) > 2.0e-6)
          throw std::runtime_error("bundle numeric policy requires represented unit keys");
        for (std::uint32_t j = 0; j < c.d_v; ++j) {
          const std::size_t vi = (static_cast<std::size_t>(t) * c.B + b) * c.d_v + j;
          if (!std::isfinite(c.values[vi]) || !std::isfinite(c.expected_outputs[vi]))
            throw std::runtime_error("live value/reference output contains NaN or Inf");
        }
      } else {
        if (!std::isnan(c.alpha[gate]) || !std::isnan(c.beta[gate]))
          throw std::runtime_error("padded gates must carry NaN sentinels");
        for (std::uint32_t j = 0; j < c.d_k; ++j) {
          const std::size_t index = (static_cast<std::size_t>(t) * c.B + b) * c.d_k + j;
          if (!std::isnan(c.keys[index]) || !std::isnan(c.queries[index]))
            throw std::runtime_error("padded key/query must carry NaN sentinels");
        }
        for (std::uint32_t j = 0; j < c.d_v; ++j) {
          const std::size_t index = (static_cast<std::size_t>(t) * c.B + b) * c.d_v + j;
          if (!std::isnan(c.values[index]) || !std::isnan(c.expected_outputs[index]))
            throw std::runtime_error("padded value/reference output must carry NaN sentinels");
        }
      }
    }
  }
  for (double x : c.expected_final_states)
    if (!std::isfinite(x)) throw std::runtime_error("non-finite FP64 final-state reference");
  for (std::uint32_t b = 0; b < c.B; ++b) {
    if (c.lengths[b] == 0) {
      for (std::size_t i = 0; i < state_per_sequence; ++i) {
        const double expected = c.initial_states[static_cast<std::size_t>(b) * state_per_sequence + i];
        const double recorded = c.expected_final_states[static_cast<std::size_t>(b) * state_per_sequence + i];
        if (expected != recorded)
          throw std::runtime_error("zero-length sequence must preserve its initial state");
      }
    }
  }
}

}  // namespace kwv08
