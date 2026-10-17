#pragma once

// Host-only reader for the deterministic KWV-07 binary input/reference bundle.
// This header does not depend on CUDA and can be exercised with a CPU compiler.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kwv07 {

enum class Kind : std::uint32_t { Grouped = 1, W4A16 = 2 };

struct Header {
  Kind kind{};
  std::uint32_t rows_or_m = 0;
  std::uint32_t k = 0;
  std::uint32_t n = 0;
  std::uint32_t experts = 0;
  std::uint32_t group_size = 0;
  std::uint32_t assignment_count = 0;
  std::uint32_t auxiliary_count = 0;  // scales for input; outputs for oracle
};

struct Assignment {
  std::int32_t row_id = 0;
  std::int32_t expert_id = 0;
  std::int32_t output_position = 0;
};

struct Input {
  Header header;
  std::vector<std::uint16_t> x_or_activation_bits;
  std::vector<std::uint16_t> up_bits;
  std::vector<std::uint16_t> gate_bits;
  std::vector<Assignment> assignments;
  std::vector<std::uint8_t> packed_codes;
  std::vector<float> scales;
};

struct Oracle {
  Header header;
  std::vector<double> output;
  std::vector<double> up_reference;
  std::vector<double> gate_reference;
  std::vector<double> sum_abs_up;
  std::vector<double> sum_abs_gate;
  std::vector<double> sum_abs_products;
};

inline std::uint32_t read_u32_le(std::istream& in) {
  std::uint32_t value = 0;
  for (unsigned shift = 0; shift != 32; shift += 8) {
    const int byte = in.get();
    if (byte == std::char_traits<char>::eof())
      throw std::runtime_error("truncated KWV-07 integer field");
    value |= static_cast<std::uint32_t>(static_cast<unsigned char>(byte)) << shift;
  }
  return value;
}

inline std::uint64_t checked_product(std::uint64_t a, std::uint64_t b,
                                     const char* label) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
    throw std::runtime_error(std::string("overflow in ") + label);
  return a * b;
}

inline Header read_header(std::istream& in, const char (&expected_magic)[9]) {
  char magic[8]{};
  in.read(magic, sizeof(magic));
  if (!in || std::string(magic, sizeof(magic)) != std::string(expected_magic, 8))
    throw std::runtime_error("bad KWV-07 binary magic");
  Header h;
  h.kind = static_cast<Kind>(read_u32_le(in));
  h.rows_or_m = read_u32_le(in);
  h.k = read_u32_le(in);
  h.n = read_u32_le(in);
  h.experts = read_u32_le(in);
  h.group_size = read_u32_le(in);
  h.assignment_count = read_u32_le(in);
  h.auxiliary_count = read_u32_le(in);
  if (h.rows_or_m == 0 || h.k == 0 || h.n == 0)
    throw std::runtime_error("KWV-07 binary header has a zero matrix dimension");
  return h;
}

inline std::vector<std::uint16_t> read_half_bits(std::istream& in,
                                                 std::uint64_t count) {
  std::vector<std::uint16_t> values(static_cast<std::size_t>(count));
  for (auto& value : values) {
    const int lo = in.get(), hi = in.get();
    if (lo == std::char_traits<char>::eof() || hi == std::char_traits<char>::eof())
      throw std::runtime_error("truncated KWV-07 FP16 array");
    value = static_cast<std::uint16_t>(static_cast<unsigned char>(lo) |
              (static_cast<unsigned>(static_cast<unsigned char>(hi)) << 8));
    if (((value >> 10) & 0x1fu) == 0x1fu)
      throw std::runtime_error("non-finite KWV-07 FP16 input");
  }
  return values;
}

inline std::vector<std::int32_t> read_i32(std::istream& in, std::uint64_t count) {
  std::vector<std::int32_t> values(static_cast<std::size_t>(count));
  for (auto& value : values) value = static_cast<std::int32_t>(read_u32_le(in));
  return values;
}

inline float read_f32_le(std::istream& in) {
  const std::uint32_t bits = read_u32_le(in);
  float value = 0.0f;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float representation");
  __builtin_memcpy(&value, &bits, sizeof(value));
  if (!std::isfinite(value)) throw std::runtime_error("non-finite KWV-07 FP32 scale");
  return value;
}

inline double read_f64_le(std::istream& in) {
  std::uint64_t bits = 0;
  for (unsigned shift = 0; shift != 64; shift += 8) {
    const int byte = in.get();
    if (byte == std::char_traits<char>::eof())
      throw std::runtime_error("truncated KWV-07 FP64 array");
    bits |= static_cast<std::uint64_t>(static_cast<unsigned char>(byte)) << shift;
  }
  double value = 0.0;
  __builtin_memcpy(&value, &bits, sizeof(value));
  if (!std::isfinite(value)) throw std::runtime_error("non-finite KWV-07 FP64 oracle value");
  return value;
}

inline void require_eof(std::istream& in) {
  if (in.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("trailing bytes in KWV-07 binary record");
}

inline Input read_input(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open KWV-07 input: " + path);
  Input result;
  result.header = read_header(in, "KWV07I01");
  const auto& h = result.header;
  if (h.kind == Kind::Grouped) {
    if (h.experts == 0 || h.group_size != 0 || h.auxiliary_count != 0 ||
        h.assignment_count == 0)
      throw std::runtime_error("invalid grouped input header");
    const std::uint64_t x_count = checked_product(h.rows_or_m, h.k, "X extent");
    const std::uint64_t w_count = checked_product(checked_product(h.experts, h.k, "weight extent"),
                                                   h.n, "weight extent");
    result.x_or_activation_bits = read_half_bits(in, x_count);
    result.up_bits = read_half_bits(in, w_count);
    result.gate_bits = read_half_bits(in, w_count);
    const auto raw_assignments = read_i32(in, checked_product(h.assignment_count, 3, "assignment extent"));
    result.assignments.reserve(h.assignment_count);
    std::vector<bool> output_seen(h.assignment_count, false);
    for (std::uint32_t i = 0; i < h.assignment_count; ++i) {
      Assignment assignment{raw_assignments[3 * i], raw_assignments[3 * i + 1],
                            raw_assignments[3 * i + 2]};
      if (assignment.row_id < 0 || static_cast<std::uint32_t>(assignment.row_id) >= h.rows_or_m ||
          assignment.expert_id < 0 || static_cast<std::uint32_t>(assignment.expert_id) >= h.experts ||
          assignment.output_position < 0 ||
          static_cast<std::uint32_t>(assignment.output_position) >= h.assignment_count ||
          output_seen[assignment.output_position])
        throw std::runtime_error("invalid or duplicate grouped assignment identity/position");
      output_seen[assignment.output_position] = true;
      result.assignments.push_back(assignment);
    }
  } else if (h.kind == Kind::W4A16) {
    if (h.experts != 0 || h.assignment_count != 0 || h.group_size == 0 ||
        h.auxiliary_count == 0)
      throw std::runtime_error("invalid W4A16 input header");
    const std::uint64_t a_count = checked_product(h.rows_or_m, h.k, "activation extent");
    const std::uint64_t w_count = checked_product(h.k, h.n, "packed weight extent");
    result.x_or_activation_bits = read_half_bits(in, a_count);
    result.packed_codes.resize(static_cast<std::size_t>((w_count + 1) / 2));
    for (auto& byte : result.packed_codes) {
      const int value = in.get();
      if (value == std::char_traits<char>::eof())
        throw std::runtime_error("truncated KWV-07 packed W4A16 weights");
      byte = static_cast<std::uint8_t>(value);
    }
    result.scales.resize(h.auxiliary_count);
    for (auto& scale : result.scales) {
      scale = read_f32_le(in);
      if (scale <= 0.0f || std::abs(scale * 8.0f) > 65504.0f)
        throw std::runtime_error("W4A16 scale violates positive finite FP16 decode bounds");
    }
    const std::uint64_t expected_scales = checked_product((h.k + h.group_size - 1) / h.group_size,
                                                           h.n, "scale extent");
    if (expected_scales != h.auxiliary_count)
      throw std::runtime_error("W4A16 scale count does not match K-group shape");
  } else {
    throw std::runtime_error("unknown KWV-07 input kind");
  }
  require_eof(in);
  return result;
}

inline Oracle read_oracle(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open KWV-07 oracle: " + path);
  Oracle result;
  result.header = read_header(in, "KWV07R01");
  const auto& h = result.header;
  if (h.kind == Kind::Grouped) {
    if (h.experts == 0 || h.group_size != 0 || h.assignment_count == 0 ||
        h.auxiliary_count != checked_product(h.assignment_count, h.n, "grouped output extent"))
      throw std::runtime_error("invalid grouped oracle header");
    const std::size_t count = h.auxiliary_count;
    result.output.resize(count);
    result.up_reference.resize(count);
    result.gate_reference.resize(count);
    result.sum_abs_up.resize(count);
    result.sum_abs_gate.resize(count);
    for (auto& value : result.output) value = read_f64_le(in);
    for (auto& value : result.up_reference) value = read_f64_le(in);
    for (auto& value : result.gate_reference) value = read_f64_le(in);
    for (auto& value : result.sum_abs_up) value = read_f64_le(in);
    for (auto& value : result.sum_abs_gate) value = read_f64_le(in);
  } else if (h.kind == Kind::W4A16) {
    if (h.experts != 0 || h.assignment_count != 0 || h.group_size == 0 ||
        h.auxiliary_count != checked_product(h.rows_or_m, h.n, "W4 output extent"))
      throw std::runtime_error("invalid W4A16 oracle header");
    const std::size_t count = h.auxiliary_count;
    result.output.resize(count);
    result.sum_abs_products.resize(count);
    for (auto& value : result.output) value = read_f64_le(in);
    for (auto& value : result.sum_abs_products) value = read_f64_le(in);
  } else {
    throw std::runtime_error("unknown KWV-07 oracle kind");
  }
  require_eof(in);
  return result;
}

inline void require_matching(const Input& input, const Oracle& oracle) {
  const auto& a = input.header;
  const auto& b = oracle.header;
  if (a.kind != b.kind || a.rows_or_m != b.rows_or_m || a.k != b.k ||
      a.n != b.n || a.experts != b.experts || a.group_size != b.group_size ||
      a.assignment_count != b.assignment_count)
    throw std::runtime_error("input and FP64 oracle headers do not match");
}

}  // namespace kwv07
