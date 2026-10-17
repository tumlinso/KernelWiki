#include "include/kwv07_bundle.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_bytes(const std::string& path, const std::vector<unsigned char>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!out) throw std::runtime_error("cannot write test record");
}

std::vector<unsigned char> read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read test input");
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void must_reject(const std::string& path) {
  try {
    (void)kwv07::read_input(path);
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("malformed input was accepted: " + path);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: test_bundle_reader_errors BUNDLE_DIR");
    const std::string base = argv[1];
    const std::string source = base + "/inputs/tiny_balanced.bin";
    auto bytes = read_bytes(source);
    const std::string bad_magic = "/tmp/kwv07-bad-magic.bin";
    const std::string truncated = "/tmp/kwv07-truncated.bin";
    const std::string trailing = "/tmp/kwv07-trailing.bin";
    const std::string duplicate_position = "/tmp/kwv07-duplicate-position.bin";
    const std::string nonfinite_half = "/tmp/kwv07-nonfinite-half.bin";

    auto changed = bytes;
    changed[0] = 'X';
    write_bytes(bad_magic, changed);
    must_reject(bad_magic);

    write_bytes(truncated, std::vector<unsigned char>(bytes.begin(), bytes.begin() + 21));
    must_reject(truncated);

    changed = bytes;
    changed[40] = 0x00;  // First FP16 X element -> +infinity (0x7c00).
    changed[41] = 0x7c;
    write_bytes(nonfinite_half, changed);
    must_reject(nonfinite_half);

    changed = bytes;
    changed.push_back(0);
    write_bytes(trailing, changed);
    must_reject(trailing);

    // tiny_balanced has rows=4,K=64,N=32,experts=4. Assignment triples
    // follow the three FP16 arrays. Repeat the first output slot in the
    // second record; the native reader must reject ambiguous scatter.
    constexpr std::size_t kHeader = 40;
    constexpr std::size_t kXBytes = 4 * 64 * 2;
    constexpr std::size_t kOneWeightBytes = 4 * 64 * 32 * 2;
    constexpr std::size_t kAssignments = kHeader + kXBytes + 2 * kOneWeightBytes;
    const std::uint32_t first_position =
        static_cast<std::uint32_t>(bytes[kAssignments + 8]) |
        (static_cast<std::uint32_t>(bytes[kAssignments + 9]) << 8) |
        (static_cast<std::uint32_t>(bytes[kAssignments + 10]) << 16) |
        (static_cast<std::uint32_t>(bytes[kAssignments + 11]) << 24);
    for (unsigned byte = 0; byte != 4; ++byte)
      bytes[kAssignments + 12 + 8 + byte] =
          static_cast<unsigned char>((first_position >> (8 * byte)) & 0xffu);
    write_bytes(duplicate_position, bytes);
    must_reject(duplicate_position);

    std::remove(bad_magic.c_str());
    std::remove(truncated.c_str());
    std::remove(trailing.c_str());
    std::remove(duplicate_position.c_str());
    std::remove(nonfinite_half.c_str());
    std::cout << "CPU bundle reader malformed-record tests PASS: 5\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CPU bundle reader malformed-record tests FAIL: " << error.what() << '\n';
    return 2;
  }
}
