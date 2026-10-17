#include "bundle_contract.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<std::uint8_t> read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read test bundle");
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
}
void write_bytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write test bundle");
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}
}

int main(int argc, char** argv) {
  if (kwv08::sha256_hex(reinterpret_cast<const std::uint8_t*>("abc"), 3) !=
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
    std::cerr << "SHA-256 CPU self-test failed\n";
    return 1;
  }
  if (argc != 2) {
    std::cerr << "usage: test_bundle_reader <bundle.bin>\n";
    return 64;
  }
  const auto original = read_bytes(argv[1]);
  const auto bundle = kwv08::read_bundle(argv[1]);
  if (bundle.cases.size() != 3 || bundle.file_sha256 != kwv08::sha256_hex(original)) {
    std::cerr << "bundle case count/file digest mismatch\n";
    return 2;
  }
  const auto& decode = bundle.cases[0];
  const auto& short_case = bundle.cases[1];
  const auto& long_case = bundle.cases[2];
  if (decode.id != "decode_many_sequences" || decode.T != 1 || decode.d_v != 32 ||
      short_case.id != "short_prefill_many_sequences" || short_case.T != 33 ||
      long_case.id != "long_prefill_many_sequences" || long_case.T != 257 ||
      long_case.d_v != 64 || long_case.d_k != 64) {
    std::cerr << "bundle shape or case order mismatch\n";
    return 3;
  }
  bool found_zero = false, found_alpha_zero = false, found_beta_zero = false;
  bool found_near_one = false, found_nan_padding = false;
  for (const auto& c : bundle.cases) {
    for (std::uint32_t b = 0; b < c.B; ++b) {
      if (c.lengths[b] == 0) found_zero = true;
      for (int t = 0; t < c.lengths[b]; ++t) {
        const std::size_t i = static_cast<std::size_t>(t) * c.B + b;
        found_alpha_zero |= c.alpha[i] == 0.0f;
        found_beta_zero |= c.beta[i] == 0.0f;
        found_near_one |= c.alpha[i] > 0.999f && c.beta[i] > 0.999f;
      }
      if (c.lengths[b] < static_cast<std::int32_t>(c.T)) {
        const std::size_t i = static_cast<std::size_t>(c.lengths[b]) * c.B + b;
        if (i < c.alpha.size()) found_nan_padding |= std::isnan(c.alpha[i]);
      }
    }
  }
  if (!found_zero || !found_alpha_zero || !found_beta_zero || !found_near_one ||
      !found_nan_padding) {
    std::cerr << "bundle semantic sentinels are missing\n";
    return 4;
  }

  auto corrupt = original;
  corrupt.back() ^= 1u;
  const std::string corrupt_path = "/tmp/kwv08-corrupt-bundle.bin";
  write_bytes(corrupt_path, corrupt);
  bool rejected = false;
  try { (void)kwv08::read_bundle(corrupt_path); }
  catch (const std::exception&) { rejected = true; }
  std::remove(corrupt_path.c_str());
  if (!rejected) {
    std::cerr << "corrupt payload was accepted\n";
    return 5;
  }

  std::cout << "bundle reader CPU checks passed; cases=" << bundle.cases.size()
            << " file_sha256=" << bundle.file_sha256
            << " payload_sha256=" << bundle.payload_sha256 << '\n';
  return 0;
}
