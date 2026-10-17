#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kwv_gemm {
namespace {

__device__ unsigned pack_half2(__half low, __half high) {
  return static_cast<unsigned>(__half_as_ushort(low)) |
         (static_cast<unsigned>(__half_as_ushort(high)) << 16);
}

// This is the Atlas E09 lane-map kernel reduced to its exact PTX instruction
// and raw pre-canonicalization store. It gates only the tested m8n8k4 row/col
// FP32-accumulator form; it says nothing about WMMA fragment ownership.
__global__ void e09_raw_map(const __half* a, const __half* b, float* raw,
                            int* raw_lane, int* raw_slot) {
  const int lane = threadIdx.x & 31;
  const int arow = (lane & 3) + (lane >= 16 ? 4 : 0);
  const int bcol = (lane & 3) + (lane >= 16 ? 4 : 0);
  const unsigned a0 = pack_half2(a[arow * 4 + 0], a[arow * 4 + 1]);
  const unsigned a1 = pack_half2(a[arow * 4 + 2], a[arow * 4 + 3]);
  const unsigned b0 = pack_half2(b[0 * 8 + bcol], b[1 * 8 + bcol]);
  const unsigned b1 = pack_half2(b[2 * 8 + bcol], b[3 * 8 + bcol]);
  const int op = (lane >> 2) & 3;
  float d[8];
  #pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    const int row = (lane & 1) + (slot & 2) + (lane >= 16 ? 4 : 0);
    const int col = (slot & 4) + (lane & 2) + (slot & 1);
    d[slot] = static_cast<float>(op * 1000 + row * 8 + col + 1);
  }
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
  #pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    const int row = (lane & 1) + (slot & 2) + (lane >= 16 ? 4 : 0);
    const int col = (slot & 4) + (lane & 2) + (slot & 1);
    const int index = op * 64 + row * 8 + col;
    raw[index] = d[slot];
    raw_lane[index] = lane;
    raw_slot[index] = slot;
  }
}

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

}  // namespace

bool run_e09_mapping_smoke() {
  std::vector<__half> a(8 * 4), b(4 * 8);
  for (int row = 0; row < 8; ++row)
    for (int k = 0; k < 4; ++k)
      a[row * 4 + k] = __float2half_rn((row % 4) == k ? 1.0f : 0.0f);
  for (int k = 0; k < 4; ++k)
    for (int col = 0; col < 8; ++col)
      b[k * 8 + col] = __float2half_rn(static_cast<float>(k * 8 + col + 1));

  __half *device_a = nullptr, *device_b = nullptr;
  float* device_raw = nullptr;
  int *device_lane = nullptr, *device_slot = nullptr;
  cuda_check(cudaMalloc(&device_a, a.size() * sizeof(__half)), "cudaMalloc A");
  cuda_check(cudaMalloc(&device_b, b.size() * sizeof(__half)), "cudaMalloc B");
  cuda_check(cudaMalloc(&device_raw, 256 * sizeof(float)), "cudaMalloc raw C");
  cuda_check(cudaMalloc(&device_lane, 256 * sizeof(int)), "cudaMalloc lane labels");
  cuda_check(cudaMalloc(&device_slot, 256 * sizeof(int)), "cudaMalloc slot labels");
  try {
    cuda_check(cudaMemcpy(device_a, a.data(), a.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy A");
    cuda_check(cudaMemcpy(device_b, b.data(), b.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy B");
    e09_raw_map<<<1, 32>>>(device_a, device_b, device_raw, device_lane, device_slot);
    cuda_check(cudaGetLastError(), "launch E09 mapping smoke");
    cuda_check(cudaDeviceSynchronize(), "synchronize E09 mapping smoke");
    std::array<float, 256> raw{};
    std::array<int, 256> lanes{}, slots{};
    cuda_check(cudaMemcpy(raw.data(), device_raw, sizeof(raw), cudaMemcpyDeviceToHost), "copy raw outputs");
    cuda_check(cudaMemcpy(lanes.data(), device_lane, sizeof(lanes), cudaMemcpyDeviceToHost), "copy lane labels");
    cuda_check(cudaMemcpy(slots.data(), device_slot, sizeof(slots), cudaMemcpyDeviceToHost), "copy slot labels");

    std::array<bool, 256> seen{};
    double max_error = 0.0;
    bool labels_valid = true;
    for (int lane = 0; lane < 32; ++lane) {
      for (int slot = 0; slot < 8; ++slot) {
        const int op = (lane >> 2) & 3;
        const int row = (lane & 1) + (slot & 2) + (lane >= 16 ? 4 : 0);
        const int col = (slot & 4) + (lane & 2) + (slot & 1);
        const int index = op * 64 + row * 8 + col;
        if (seen[index] || lanes[index] != lane || slots[index] != slot) labels_valid = false;
        seen[index] = true;
      }
    }
    for (int op = 0; op < 4; ++op) {
      for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
          const int index = op * 64 + row * 8 + col;
          const double expected = op * 1000 + row * 8 + col + 1 + (row % 4) * 8 + col + 1;
          max_error = std::max(max_error, std::abs(static_cast<double>(raw[index]) - expected));
        }
      }
    }
    const bool full_coverage = std::all_of(seen.begin(), seen.end(), [](bool x) { return x; });
    std::cout << "{\"name\":\"e09_m8n8k4_row_col_fp32\",\"coordinates\":256"
              << ",\"max_abs_error\":" << max_error
              << ",\"labels_valid\":" << (labels_valid ? "true" : "false")
              << ",\"full_coverage\":" << (full_coverage ? "true" : "false") << "}\n";
    cudaFree(device_slot); cudaFree(device_lane); cudaFree(device_raw);
    cudaFree(device_b); cudaFree(device_a);
    return labels_valid && full_coverage && max_error == 0.0;
  } catch (...) {
    cudaFree(device_slot); cudaFree(device_lane); cudaFree(device_raw);
    cudaFree(device_b); cudaFree(device_a);
    throw;
  }
}

}  // namespace kwv_gemm
