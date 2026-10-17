#include "gemm_contract.hpp"

#include <cstdlib>
#include <iostream>

namespace {

std::vector<float> read_output(const std::string& path, std::size_t count) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open FP32 output: " + path);
  std::vector<float> result(count);
  for (auto& value : result) value = kwv_gemm::read_f32_le(in);
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing bytes in FP32 output");
  return result;
}

void oracle_self_test(const std::string& case_id) {
  double max_error = 0.0, reference_max_abs = 0.0;
  double sum_abs_error = 0.0, sum_abs_reference_max = 0.0;
  const bool passed = kwv_gemm::oracle_self_test(
      max_error, reference_max_abs, sum_abs_error, sum_abs_reference_max);
  std::cout << std::setprecision(17)
      << "{\"format\":\"KERNELWIKI-CPU-ORACLE/1\",\"case_id\":\"" << case_id
      << "\",\"passed\":" << (passed ? "true" : "false")
      << ",\"checks\":[{\"name\":\"exact_fp64_2x2_core_self_test\",\"max_abs_error\":"
      << max_error << ",\"reference_max_abs\":" << reference_max_abs
      << ",\"atol\":0,\"rtol\":0},{\"name\":\"exact_fp64_2x2_sum_abs_products_self_test\",\"max_abs_error\":"
      << sum_abs_error << ",\"reference_max_abs\":" << sum_abs_reference_max
      << ",\"atol\":0,\"rtol\":0}] }\n";
  if (!passed) std::exit(2);
}

void generate_reference(const std::string& case_id, const std::string& input_path,
                       const std::string& reference_path) {
  const auto shape = kwv_gemm::shape_by_id(case_id);
  const auto inputs = kwv_gemm::read_inputs(shape, input_path);
  const auto oracle = kwv_gemm::compute_oracle(inputs);
  kwv_gemm::write_oracle_file(shape, oracle, reference_path);
}

void compare(const std::string& case_id, const std::string& operator_name,
             const std::string& input_path, const std::string& reference_path,
             const std::string& output_path) {
  const auto shape = kwv_gemm::shape_by_id(case_id);
  const auto inputs = kwv_gemm::read_inputs(shape, input_path);
  const auto oracle = kwv_gemm::read_oracle_file(shape, reference_path);
  const bool bias_relu = operator_name == "gemm_bias_relu";
  if (!bias_relu && operator_name != "gemm")
    throw std::invalid_argument("operator must be gemm or gemm_bias_relu");
  const auto output = read_output(output_path, static_cast<std::size_t>(shape.m) * shape.n);
  const auto check = kwv_gemm::check_output(inputs, oracle, output, bias_relu);
  kwv_gemm::write_check_json(std::cout, shape, check, bias_relu);
  if (check.failed_outputs != 0) std::exit(3);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "generate") {
      const auto shape = kwv_gemm::shape_by_id(argv[2]);
      kwv_gemm::write_inputs(kwv_gemm::make_inputs(shape), argv[3]);
      return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "generate-reference") {
      generate_reference(argv[2], argv[3], argv[4]);
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "oracle-self-test") {
      oracle_self_test(argv[2]);
      return 0;
    }
    if (argc == 7 && std::string(argv[1]) == "check-output") {
      compare(argv[2], argv[3], argv[4], argv[5], argv[6]);
      return 0;
    }
    std::cerr << "usage:\n"
              << "  gemm_cpu_oracle generate CASE INPUT.bin\n"
              << "  gemm_cpu_oracle generate-reference CASE INPUT.bin REFERENCE.f64le\n"
              << "  gemm_cpu_oracle oracle-self-test CASE_ID\n"
              << "  gemm_cpu_oracle check-output CASE gemm|gemm_bias_relu INPUT.bin REFERENCE.f64le OUTPUT.f32le\n";
    return 64;
  } catch (const std::exception& error) {
    std::cerr << "gemm_cpu_oracle: " << error.what() << '\n';
    return 1;
  }
}
