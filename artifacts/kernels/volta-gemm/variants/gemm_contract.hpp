#pragma once

// Host-only contract and FP64 oracle for the KWV-04 draft.  The only input to
// the oracle is the serialized binary16/FP32 representation file; it never
// consults pre-conversion random values or a device implementation.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kwv_gemm {

struct Shape {
  const char* id;
  std::uint32_t m;
  std::uint32_t n;
  std::uint32_t k;
  std::uint32_t seed;
};

inline Shape shape_by_id(const std::string& id) {
  if (id == "large_dense") return {"large_dense", 1024, 1024, 1024, 0x70a001u};
  if (id == "small_m") return {"small_m", 16, 4096, 4096, 0x70a002u};
  if (id == "ragged") return {"ragged", 127, 513, 769, 0x70a003u};
  throw std::invalid_argument("unknown GEMM case: " + id);
}

struct Inputs {
  Shape shape{};
  std::vector<std::uint16_t> a_bits;
  std::vector<std::uint16_t> b_bits;
  std::vector<float> bias;
};

struct Oracle {
  std::vector<double> core;
  std::vector<double> sum_abs_products;
};

struct Check {
  double max_abs_error = 0.0;
  double max_admission_tolerance = 0.0;
  double max_tolerance_ratio = 0.0;
  std::uint64_t checked_outputs = 0;
  std::uint64_t failed_outputs = 0;
  std::uint64_t nonfinite_outputs = 0;
};

// The generated values are q * 2^-fractional_bits, for |q| <= 128.  They are
// exactly representable normal binary16 values, encoded without CUDA headers.
inline std::uint16_t exact_quantized_half(int q, unsigned fractional_bits) {
  const bool negative = q < 0;
  const unsigned magnitude = static_cast<unsigned>(negative ? -q : q);
  if (magnitude == 0) return 0;
  unsigned top = 0;
  for (unsigned x = magnitude; x >>= 1;) ++top;
  const int exponent = static_cast<int>(top) - static_cast<int>(fractional_bits);
  if (exponent < -14 || exponent > 15 || top > 10)
    throw std::invalid_argument("quantized value is outside the normal binary16 range");
  const unsigned significand_tail = magnitude - (1u << top);
  const unsigned fraction = significand_tail << (10u - top);
  const unsigned sign = negative ? 0x8000u : 0u;
  return static_cast<std::uint16_t>(sign | (static_cast<unsigned>(exponent + 15) << 10) | fraction);
}

inline double half_to_double(std::uint16_t bits) {
  const bool negative = (bits & 0x8000u) != 0;
  const unsigned exponent = (bits >> 10) & 0x1fu;
  const unsigned fraction = bits & 0x3ffu;
  double value = 0.0;
  if (exponent == 0) {
    value = std::ldexp(static_cast<double>(fraction), -24);
  } else if (exponent == 31) {
    value = fraction == 0 ? std::numeric_limits<double>::infinity()
                          : std::numeric_limits<double>::quiet_NaN();
  } else {
    value = std::ldexp(static_cast<double>(1024u + fraction),
                       static_cast<int>(exponent) - 25);
  }
  return negative ? -value : value;
}

inline float half_to_float(std::uint16_t bits) {
  return static_cast<float>(half_to_double(bits));
}

inline void write_u32_le(std::ostream& out, std::uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8)
    out.put(static_cast<char>((value >> shift) & 0xffu));
}

inline std::uint32_t read_u32_le(std::istream& in) {
  std::uint32_t value = 0;
  for (unsigned shift = 0; shift != 32; shift += 8) {
    const int byte = in.get();
    if (byte == std::char_traits<char>::eof()) throw std::runtime_error("truncated GEMM input");
    value |= static_cast<std::uint32_t>(static_cast<unsigned char>(byte)) << shift;
  }
  return value;
}

inline void write_u64_le(std::ostream& out, std::uint64_t value) {
  for (unsigned shift = 0; shift != 64; shift += 8)
    out.put(static_cast<char>((value >> shift) & 0xffu));
}

inline std::uint64_t read_u64_le(std::istream& in) {
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift != 64; shift += 8) {
    const int byte = in.get();
    if (byte == std::char_traits<char>::eof()) throw std::runtime_error("truncated GEMM reference");
    value |= static_cast<std::uint64_t>(static_cast<unsigned char>(byte)) << shift;
  }
  return value;
}

inline void write_f32_le(std::ostream& out, float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float representation");
  std::memcpy(&bits, &value, sizeof(bits));
  write_u32_le(out, bits);
}

inline float read_f32_le(std::istream& in) {
  const std::uint32_t bits = read_u32_le(in);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline void write_f64_le(std::ostream& out, double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected double representation");
  std::memcpy(&bits, &value, sizeof(bits));
  write_u64_le(out, bits);
}

inline double read_f64_le(std::istream& in) {
  const std::uint64_t bits = read_u64_le(in);
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline std::uint32_t xorshift32(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

inline Inputs make_inputs(const Shape& shape) {
  Inputs inputs;
  inputs.shape = shape;
  inputs.a_bits.resize(static_cast<std::size_t>(shape.m) * shape.k);
  inputs.b_bits.resize(static_cast<std::size_t>(shape.k) * shape.n);
  inputs.bias.resize(shape.n);
  std::uint32_t rng = shape.seed ? shape.seed : 1u;
  auto next_q = [&]() { return static_cast<int>(xorshift32(rng) % 257u) - 128; };
  for (auto& value : inputs.a_bits) value = exact_quantized_half(next_q(), 8);
  for (auto& value : inputs.b_bits) value = exact_quantized_half(next_q(), 8);
  for (auto& value : inputs.bias) value = static_cast<float>(next_q()) * (1.0f / 128.0f);
  return inputs;
}

inline void write_inputs(const Inputs& inputs, const std::string& path) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot create represented input: " + path);
  out.write("KWVGEMM1", 8);
  write_u32_le(out, inputs.shape.m);
  write_u32_le(out, inputs.shape.n);
  write_u32_le(out, inputs.shape.k);
  write_u32_le(out, inputs.shape.seed);
  for (const auto value : inputs.a_bits) {
    out.put(static_cast<char>(value & 0xffu));
    out.put(static_cast<char>((value >> 8) & 0xffu));
  }
  for (const auto value : inputs.b_bits) {
    out.put(static_cast<char>(value & 0xffu));
    out.put(static_cast<char>((value >> 8) & 0xffu));
  }
  for (const auto value : inputs.bias) write_f32_le(out, value);
  if (!out) throw std::runtime_error("failed while writing represented input: " + path);
}

inline Inputs read_inputs(const Shape& expected, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open represented input: " + path);
  char magic[8]{};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, "KWVGEMM1", 8) != 0) throw std::runtime_error("bad GEMM input magic");
  Inputs inputs;
  inputs.shape = {expected.id, read_u32_le(in), read_u32_le(in), read_u32_le(in), read_u32_le(in)};
  if (inputs.shape.m != expected.m || inputs.shape.n != expected.n ||
      inputs.shape.k != expected.k || inputs.shape.seed != expected.seed)
    throw std::runtime_error("GEMM input shape/seed does not match the named case");
  inputs.a_bits.resize(static_cast<std::size_t>(inputs.shape.m) * inputs.shape.k);
  inputs.b_bits.resize(static_cast<std::size_t>(inputs.shape.k) * inputs.shape.n);
  inputs.bias.resize(inputs.shape.n);
  auto read_half = [&]() {
    const int lo = in.get(), hi = in.get();
    if (lo == std::char_traits<char>::eof() || hi == std::char_traits<char>::eof())
      throw std::runtime_error("truncated binary16 GEMM input");
    return static_cast<std::uint16_t>(static_cast<unsigned char>(lo) |
                                      (static_cast<unsigned>(static_cast<unsigned char>(hi)) << 8));
  };
  for (auto& value : inputs.a_bits) value = read_half();
  for (auto& value : inputs.b_bits) value = read_half();
  for (auto& value : inputs.bias) value = read_f32_le(in);
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing bytes in GEMM input");
  for (const auto value : inputs.a_bits)
    if (!std::isfinite(half_to_double(value))) throw std::runtime_error("non-finite A input");
  for (const auto value : inputs.b_bits)
    if (!std::isfinite(half_to_double(value))) throw std::runtime_error("non-finite B input");
  for (const auto value : inputs.bias)
    if (!std::isfinite(value)) throw std::runtime_error("non-finite bias input");
  return inputs;
}

inline Oracle compute_oracle(const Inputs& inputs) {
  const auto& s = inputs.shape;
  std::vector<double> a(inputs.a_bits.size()), b(inputs.b_bits.size());
  std::transform(inputs.a_bits.begin(), inputs.a_bits.end(), a.begin(), half_to_double);
  std::transform(inputs.b_bits.begin(), inputs.b_bits.end(), b.begin(), half_to_double);
  Oracle oracle;
  oracle.core.resize(static_cast<std::size_t>(s.m) * s.n);
  oracle.sum_abs_products.resize(oracle.core.size());
  std::vector<double> row_sum(s.n), row_sum_abs(s.n);
  for (std::uint32_t row = 0; row < s.m; ++row) {
    std::fill(row_sum.begin(), row_sum.end(), 0.0);
    std::fill(row_sum_abs.begin(), row_sum_abs.end(), 0.0);
    for (std::uint32_t inner = 0; inner < s.k; ++inner) {
      const double left = a[static_cast<std::size_t>(row) * s.k + inner];
      const std::size_t b_row = static_cast<std::size_t>(inner) * s.n;
      for (std::uint32_t col = 0; col < s.n; ++col) {
        const double product = left * b[b_row + col];
        row_sum[col] += product;
        row_sum_abs[col] += std::abs(product);
      }
    }
    const std::size_t output_row = static_cast<std::size_t>(row) * s.n;
    std::copy(row_sum.begin(), row_sum.end(), oracle.core.begin() + output_row);
    std::copy(row_sum_abs.begin(), row_sum_abs.end(), oracle.sum_abs_products.begin() + output_row);
  }
  return oracle;
}

inline void write_oracle_file(const Shape& shape, const Oracle& oracle,
                              const std::string& path) {
  const std::size_t count = static_cast<std::size_t>(shape.m) * shape.n;
  if (oracle.core.size() != count || oracle.sum_abs_products.size() != count)
    throw std::invalid_argument("FP64 reference extent does not match GEMM shape");
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot create CPU reference: " + path);
  out.write("KWVREF01", 8);
  write_u32_le(out, shape.m);
  write_u32_le(out, shape.n);
  write_u32_le(out, shape.k);
  write_u32_le(out, shape.seed);
  for (double value : oracle.core) write_f64_le(out, value);
  for (double value : oracle.sum_abs_products) write_f64_le(out, value);
  if (!out) throw std::runtime_error("failed while writing CPU reference: " + path);
}

inline Oracle read_oracle_file(const Shape& expected, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open CPU reference: " + path);
  char magic[8]{};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, "KWVREF01", 8) != 0)
    throw std::runtime_error("bad CPU reference magic");
  const auto m = read_u32_le(in), n = read_u32_le(in), k = read_u32_le(in);
  const auto seed = read_u32_le(in);
  if (m != expected.m || n != expected.n || k != expected.k || seed != expected.seed)
    throw std::runtime_error("CPU reference shape/seed does not match the named case");
  const std::size_t count = static_cast<std::size_t>(m) * n;
  Oracle oracle;
  oracle.core.resize(count);
  oracle.sum_abs_products.resize(count);
  for (double& value : oracle.core) value = read_f64_le(in);
  for (double& value : oracle.sum_abs_products) value = read_f64_le(in);
  if (in.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("trailing bytes in CPU reference");
  for (double value : oracle.core)
    if (!std::isfinite(value)) throw std::runtime_error("non-finite CPU reference output");
  for (double value : oracle.sum_abs_products)
    if (!std::isfinite(value) || value < 0.0)
      throw std::runtime_error("invalid CPU reference magnitude");
  return oracle;
}

inline double gamma_tolerance(std::uint32_t operations, double magnitude) {
  constexpr double u = 0x1p-24;
  const double ku = static_cast<double>(operations) * u;
  if (!(ku < 1.0)) throw std::invalid_argument("operation count invalidates FP32 gamma bound");
  return (ku / (1.0 - ku)) * magnitude;
}

inline Check check_output(const Inputs& inputs, const Oracle& oracle,
                          const std::vector<float>& actual, bool bias_relu) {
  const auto& s = inputs.shape;
  const std::size_t count = static_cast<std::size_t>(s.m) * s.n;
  if (actual.size() != count || oracle.core.size() != count ||
      oracle.sum_abs_products.size() != count)
    throw std::invalid_argument("output/oracle extent does not match GEMM case");
  Check result;
  for (std::uint32_t row = 0; row < s.m; ++row) {
    for (std::uint32_t col = 0; col < s.n; ++col) {
      const auto index = static_cast<std::size_t>(row) * s.n + col;
      const double ref_core = oracle.core[index];
      const double bias = inputs.bias[col];
      const double reference = bias_relu ? std::max(0.0, ref_core + bias) : ref_core;
      const double magnitude = oracle.sum_abs_products[index] + (bias_relu ? std::abs(bias) : 0.0);
      const double tolerance = gamma_tolerance(s.k + (bias_relu ? 1u : 0u), magnitude);
      ++result.checked_outputs;
      result.max_admission_tolerance = std::max(result.max_admission_tolerance, tolerance);
      if (!std::isfinite(actual[index])) {
        ++result.nonfinite_outputs;
        ++result.failed_outputs;
        continue;
      }
      const double error = std::abs(static_cast<double>(actual[index]) - reference);
      result.max_abs_error = std::max(result.max_abs_error, error);
      const double ratio = tolerance == 0.0 ? (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                                            : error / tolerance;
      result.max_tolerance_ratio = std::max(result.max_tolerance_ratio, ratio);
      if (error > tolerance) ++result.failed_outputs;
    }
  }
  return result;
}

inline bool oracle_self_test(double& max_abs_error, double& reference_max_abs,
                             double& sum_abs_error, double& sum_abs_reference_max) {
  // Independent exact 2x2 integer product used to exercise the same FP64
  // accumulation routine that creates each immutable case reference.
  Inputs inputs;
  inputs.shape = {"self_test", 2, 2, 2, 0x70a0ffu};
  inputs.a_bits = {exact_quantized_half(1, 0), exact_quantized_half(2, 0),
                   exact_quantized_half(3, 0), exact_quantized_half(4, 0)};
  inputs.b_bits = {exact_quantized_half(5, 0), exact_quantized_half(6, 0),
                   exact_quantized_half(7, 0), exact_quantized_half(8, 0)};
  inputs.bias = {0.0f, 0.0f};
  const double expected[4]{19.0, 22.0, 43.0, 50.0};
  const Oracle oracle = compute_oracle(inputs);
  max_abs_error = 0.0;
  reference_max_abs = 0.0;
  sum_abs_error = 0.0;
  sum_abs_reference_max = 0.0;
  for (int i = 0; i < 4; ++i) {
    max_abs_error = std::max(max_abs_error, std::abs(oracle.core[i] - expected[i]));
    reference_max_abs = std::max(reference_max_abs, std::abs(expected[i]));
    sum_abs_error = std::max(sum_abs_error,
                             std::abs(oracle.sum_abs_products[i] - expected[i]));
    sum_abs_reference_max = std::max(sum_abs_reference_max, std::abs(expected[i]));
  }
  return max_abs_error == 0.0 && sum_abs_error == 0.0;
}

inline void write_check_json(std::ostream& out, const Shape& shape,
                             const Check& check, bool bias_relu,
                             const std::string& implementation = "cpu_output_check",
                             int cublas_version = 0, int shared_stages = 0) {
  out << std::setprecision(17)
      << "{\"format\":\"KERNELWIKI-GEMM-CHECK/1\",\"case_id\":\"" << shape.id
      << "\",\"operator\":\"" << (bias_relu ? "gemm_bias_relu" : "gemm")
      << "\",\"implementation\":\"" << implementation
      << "\",\"cublas_api_version\":";
  if (cublas_version > 0) out << cublas_version;
  else out << "null";
  out << ",\"shared_stages\":";
  if (shared_stages > 0) out << shared_stages;
  else out << "null";
  out
      << ",\"max_abs_error\":" << check.max_abs_error
      << ",\"max_admission_tolerance\":" << check.max_admission_tolerance
      << ",\"max_tolerance_ratio\":" << check.max_tolerance_ratio
      << ",\"checked_outputs\":" << check.checked_outputs
      << ",\"failed_outputs\":" << check.failed_outputs
      << ",\"nonfinite_outputs\":" << check.nonfinite_outputs
      << ",\"passed\":" << (check.failed_outputs == 0 ? "true" : "false") << "}\n";
}

}  // namespace kwv_gemm
