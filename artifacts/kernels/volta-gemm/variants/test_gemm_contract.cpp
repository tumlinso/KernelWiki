#include "gemm_contract.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace kwv_gemm;
  assert(half_to_double(exact_quantized_half(8, 3)) == 1.0);
  assert(half_to_double(exact_quantized_half(-4, 3)) == -0.5);
  assert(half_to_double(exact_quantized_half(0, 8)) == 0.0);

  Inputs inputs;
  inputs.shape = {"tiny", 2, 2, 3, 0x70a004u};
  inputs.a_bits = {exact_quantized_half(8, 3), exact_quantized_half(4, 3),
                   exact_quantized_half(16, 3), exact_quantized_half(-8, 3),
                   exact_quantized_half(4, 3), exact_quantized_half(8, 3)};
  inputs.b_bits = {exact_quantized_half(16, 3), exact_quantized_half(-8, 3),
                   exact_quantized_half(8, 3), exact_quantized_half(16, 3),
                   exact_quantized_half(4, 3), exact_quantized_half(8, 3)};
  inputs.bias = {0.25f, -0.5f};

  const Oracle oracle = compute_oracle(inputs);
  const double expected[4]{3.5, 2.0, -1.0, 3.0};
  for (int i = 0; i < 4; ++i) assert(oracle.core[i] == expected[i]);
  std::vector<float> core(4);
  for (int i = 0; i < 4; ++i) core[i] = static_cast<float>(oracle.core[i]);
  assert(check_output(inputs, oracle, core, false).failed_outputs == 0);

  const std::vector<float> fused{3.75f, 1.5f, 0.0f, 2.5f};
  assert(check_output(inputs, oracle, fused, true).failed_outputs == 0);
  auto wrong = core;
  wrong[0] += 1.0f;
  assert(check_output(inputs, oracle, wrong, false).failed_outputs != 0);

  const char* reference_path = "/tmp/kwv04-gemm-reference-test.bin";
  write_oracle_file(inputs.shape, oracle, reference_path);
  const Oracle restored_oracle = read_oracle_file(inputs.shape, reference_path);
  assert(restored_oracle.core == oracle.core);
  assert(restored_oracle.sum_abs_products == oracle.sum_abs_products);
  bool rejected_wrong_shape = false;
  try {
    (void)read_oracle_file(Shape{"wrong", 2, 2, 4, inputs.shape.seed}, reference_path);
  } catch (const std::runtime_error&) {
    rejected_wrong_shape = true;
  }
  assert(rejected_wrong_shape);
  {
    std::ofstream append(reference_path, std::ios::binary | std::ios::app);
    append.put('x');
  }
  bool rejected_trailing_byte = false;
  try {
    (void)read_oracle_file(inputs.shape, reference_path);
  } catch (const std::runtime_error&) {
    rejected_trailing_byte = true;
  }
  assert(rejected_trailing_byte);
  std::remove(reference_path);

  const auto shape = shape_by_id("ragged");
  const auto generated = make_inputs(shape);
  const char* path = "/tmp/kwv04-gemm-contract-test.bin";
  write_inputs(generated, path);
  const auto restored = read_inputs(shape, path);
  assert(restored.a_bits == generated.a_bits);
  assert(restored.b_bits == generated.b_bits);
  assert(restored.bias == generated.bias);
  std::remove(path);

  double self_error = 0.0, self_reference = 0.0;
  double sum_abs_error = 0.0, sum_abs_reference = 0.0;
  assert(oracle_self_test(self_error, self_reference, sum_abs_error, sum_abs_reference));
  assert(self_error == 0.0 && self_reference == 50.0);
  assert(sum_abs_error == 0.0 && sum_abs_reference == 50.0);
  std::cout << "gemm contract CPU checks passed\n";
  return 0;
}
