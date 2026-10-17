#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace kwv_grouped {

struct Assignment {
  std::int32_t row_id;
  std::int32_t expert_id;
  std::int32_t output_position;
};

// Small-M fused Q1 candidate. One CTA owns one assignment and one N tile;
// X[row] is staged once and reused by paired FP32 gate/up dot accumulators.
// The candidate is SIMT and intentionally bounded to the small-M regime.
cudaError_t launch_gated_pair_small_m(
    const std::uint16_t* x_bits, const std::uint16_t* up_bits,
    const std::uint16_t* gate_bits, const Assignment* assignments,
    float* output, int rows, int experts, int assignment_count,
    int k, int n, cudaStream_t stream);

// Decode the exact signed-nibble format to represented FP16 [K,N].
cudaError_t launch_decode_w4a16(
    const std::uint8_t* packed_codes, const float* scales,
    std::uint16_t* decoded_bits, int k, int n, int group_size,
    cudaStream_t stream);

// Small-M direct decode-and-dot W4A16 candidate. It recomputes decoded
// weights for each activation use; no persistent decoded cache is implied.
cudaError_t launch_w4a16_simt_small_m(
    const std::uint16_t* activation_bits, const std::uint8_t* packed_codes,
    const float* scales, float* output, int m, int k, int n,
    int group_size, cudaStream_t stream);

}  // namespace kwv_grouped
