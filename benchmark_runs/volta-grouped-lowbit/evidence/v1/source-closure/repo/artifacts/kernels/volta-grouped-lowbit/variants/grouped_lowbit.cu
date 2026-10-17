#include "grouped_lowbit.cuh"

#include <cuda_fp16.h>
#include <cstddef>

namespace kwv_grouped {
namespace {

__device__ __forceinline__ int signed_code(const std::uint8_t* packed,
                                           std::size_t index) {
  const std::uint8_t byte = packed[index >> 1];
  const int nibble = static_cast<int>((byte >> ((index & 1u) ? 4 : 0)) & 0x0fu);
  return nibble < 8 ? nibble : nibble - 16;
}

__global__ void gated_pair_small_m_kernel(
    const std::uint16_t* x_bits, const std::uint16_t* up_bits,
    const std::uint16_t* gate_bits, const Assignment* assignments,
    float* output, int rows, int experts, int assignment_count,
    int k, int n) {
  extern __shared__ __half x_row[];
  const int assignment_index = blockIdx.y;
  if (assignment_index >= assignment_count) return;
  const Assignment assignment = assignments[assignment_index];
  if (assignment.row_id < 0 || assignment.row_id >= rows ||
      assignment.expert_id < 0 || assignment.expert_id >= experts ||
      assignment.output_position < 0 || assignment.output_position >= assignment_count)
    return;

  const std::size_t x_base = static_cast<std::size_t>(assignment.row_id) * k;
  for (int kk = threadIdx.x; kk < k; kk += blockDim.x)
    x_row[kk] = __ushort_as_half(x_bits[x_base + kk]);
  __syncthreads();

  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n) return;
  const std::size_t weight_base =
      static_cast<std::size_t>(assignment.expert_id) * k * n + col;
  float up = 0.0f;
  float gate = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    const float x = __half2float(x_row[kk]);
    const std::size_t index = static_cast<std::size_t>(kk) * n;
    const float wu = __half2float(__ushort_as_half(up_bits[weight_base + index]));
    const float wg = __half2float(__ushort_as_half(gate_bits[weight_base + index]));
    up = fmaf(x, wu, up);
    gate = fmaf(x, wg, gate);
  }
  // Contracted operator: SiLU(gate) * up, with FP32 projections and output.
  const float silu = gate / (1.0f + expf(-gate));
  output[static_cast<std::size_t>(assignment.output_position) * n + col] = silu * up;
}

__global__ void decode_w4a16_kernel(
    const std::uint8_t* packed_codes, const float* scales,
    std::uint16_t* decoded_bits, int k, int n, int group_size) {
  const std::size_t count = static_cast<std::size_t>(k) * n;
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int row = static_cast<int>(index / n);
  const int col = static_cast<int>(index % n);
  const int group = row / group_size;
  const int q = signed_code(packed_codes, index);
  const float scaled = scales[static_cast<std::size_t>(group) * n + col] * static_cast<float>(q);
  const __half represented = __float2half_rn(scaled);
  decoded_bits[index] = __half_as_ushort(represented);
}

__global__ void w4a16_simt_small_m_kernel(
    const std::uint16_t* activation_bits, const std::uint8_t* packed_codes,
    const float* scales, float* output, int m, int k, int n,
    int group_size) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (row >= m || col >= n) return;
  float sum = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    const std::size_t weight_index = static_cast<std::size_t>(kk) * n + col;
    const int q = signed_code(packed_codes, weight_index);
    const int group = kk / group_size;
    const float scale = scales[static_cast<std::size_t>(group) * n + col];
    const __half w_half = __float2half_rn(scale * static_cast<float>(q));
    const float x = __half2float(__ushort_as_half(
        activation_bits[static_cast<std::size_t>(row) * k + kk]));
    sum = fmaf(x, __half2float(w_half), sum);
  }
  output[static_cast<std::size_t>(row) * n + col] = sum;
}

}  // namespace

cudaError_t launch_gated_pair_small_m(
    const std::uint16_t* x_bits, const std::uint16_t* up_bits,
    const std::uint16_t* gate_bits, const Assignment* assignments,
    float* output, int rows, int experts, int assignment_count,
    int k, int n, cudaStream_t stream) {
  if (!x_bits || !up_bits || !gate_bits || !assignments || !output ||
      rows <= 0 || experts <= 0 || assignment_count <= 0 || k <= 0 || n <= 0)
    return cudaErrorInvalidValue;
  constexpr int Threads = 128;
  const dim3 grid((n + Threads - 1) / Threads, assignment_count);
  gated_pair_small_m_kernel<<<grid, Threads, static_cast<std::size_t>(k) * sizeof(__half), stream>>>(
      x_bits, up_bits, gate_bits, assignments, output, rows, experts,
      assignment_count, k, n);
  return cudaGetLastError();
}

cudaError_t launch_decode_w4a16(
    const std::uint8_t* packed_codes, const float* scales,
    std::uint16_t* decoded_bits, int k, int n, int group_size,
    cudaStream_t stream) {
  if (!packed_codes || !scales || !decoded_bits || k <= 0 || n <= 0 || group_size <= 0)
    return cudaErrorInvalidValue;
  constexpr int Threads = 256;
  const std::size_t count = static_cast<std::size_t>(k) * n;
  const int blocks = static_cast<int>((count + Threads - 1) / Threads);
  decode_w4a16_kernel<<<blocks, Threads, 0, stream>>>(
      packed_codes, scales, decoded_bits, k, n, group_size);
  return cudaGetLastError();
}

cudaError_t launch_w4a16_simt_small_m(
    const std::uint16_t* activation_bits, const std::uint8_t* packed_codes,
    const float* scales, float* output, int m, int k, int n,
    int group_size, cudaStream_t stream) {
  if (!activation_bits || !packed_codes || !scales || !output ||
      m <= 0 || k <= 0 || n <= 0 || group_size <= 0)
    return cudaErrorInvalidValue;
  constexpr int Threads = 128;
  const dim3 grid((n + Threads - 1) / Threads, m);
  w4a16_simt_small_m_kernel<<<grid, Threads, 0, stream>>>(
      activation_bits, packed_codes, scales, output, m, k, n, group_size);
  return cudaGetLastError();
}

}  // namespace kwv_grouped
