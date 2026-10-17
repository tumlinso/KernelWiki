#include "include/kwv07_bundle.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 2) throw std::invalid_argument("usage: test_bundle_reader BUNDLE_DIR [CASE_ID ...]");
    const std::string root = argv[1];
    std::vector<std::string> cases;
    if (argc > 2) {
      for (int i = 2; i < argc; ++i) cases.emplace_back(argv[i]);
    } else {
      cases = {"tiny_balanced", "tiny_skewed", "moderate_balanced",
               "moderate_skewed", "weighted_balanced", "weighted_skewed",
               "low_m", "moderate_m", "weighted_low_m", "weighted_moderate_m"};
    }
    std::size_t outputs = 0;
    for (const auto& id : cases) {
      const auto input = kwv07::read_input(root + "/inputs/" + id + ".bin");
      const auto oracle = kwv07::read_oracle(root + "/oracles/" + id + ".bin");
      kwv07::require_matching(input, oracle);
      outputs += oracle.output.size();
      std::cout << id << ": input/oracle record matches; outputs="
                << oracle.output.size() << '\n';
    }
    std::cout << "CPU bundle reader PASS: cases=" << cases.size()
              << " outputs=" << outputs << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CPU bundle reader FAIL: " << error.what() << '\n';
    return 2;
  }
}
