#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace kwv_gemm {

enum class GemmTile { Large64, SmallM16x32, Ragged32x64 };

// Launch an explicit m8n8k4 row/col FP32-accumulator kernel. This prototype
// reuses E09's tested per-lane accumulator map; it does not infer WMMA slots.
cudaError_t launch_ptx_gemm(const std::uint16_t* a_bits,
                            const std::uint16_t* b_bits,
                            const float* bias, float* output,
                            int m, int n, int k, GemmTile tile,
                            int shared_stages, bool bias_relu,
                            cudaStream_t stream);

}  // namespace kwv_gemm
