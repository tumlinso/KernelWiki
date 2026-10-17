#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr int kSide = 16;
constexpr int kCount = kSide * kSide;
constexpr int kFmaCount = 64;

void check(cudaError_t status, const char* expression, int line) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at line " << line << " for " << expression << ": "
              << cudaGetErrorString(status) << '\n';
    std::exit(2);
  }
}
#define CUDA_CHECK(expr) check((expr), #expr, __LINE__)

__device__ __forceinline__ float ptx_fma_rn(float a, float b, float c) {
  float result;
  asm volatile("fma.rn.f32 %0, %1, %2, %3;"
               : "=f"(result) : "f"(a), "f"(b), "f"(c));
  return result;
}

__global__ void inline_ptx_fma(const float* a, const float* b, const float* c,
                               float* output) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kFmaCount) output[i] = ptx_fma_rn(a[i], b[i], c[i]);
}

__global__ void wmma_16x16x16(const half* a, const half* b, float* output) {
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  wmma::load_matrix_sync(a_frag, a, kSide);
  wmma::load_matrix_sync(b_frag, b, kSide);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(output, c_frag, kSide, wmma::mem_row_major);
}

bool load_half_inputs(const std::string& path, std::vector<half>& left,
                      std::vector<half>& right) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::cerr << "cannot open represented input: " << path << '\n';
    return false;
  }
  left.resize(kCount);
  right.resize(kCount);
  input.read(reinterpret_cast<char*>(left.data()), sizeof(half) * kCount);
  input.read(reinterpret_cast<char*>(right.data()), sizeof(half) * kCount);
  if (!input || input.peek() != std::char_traits<char>::eof()) {
    std::cerr << "represented input must contain exactly two 16x16 FP16 matrices\n";
    return false;
  }
  return true;
}

bool close(float actual, double expected, double atol, double rtol) {
  return std::abs(static_cast<double>(actual) - expected) <=
         atol + rtol * std::abs(expected);
}

int self_test(const std::string& input_path) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "expected Volta SM70; observed compute capability "
              << properties.major << '.' << properties.minor << '\n';
    return 3;
  }

  std::vector<half> h_left, h_right;
  if (!load_half_inputs(input_path, h_left, h_right)) return 2;
  float h_a[kFmaCount], h_b[kFmaCount], h_c[kFmaCount], h_ptx[kFmaCount]{};
  for (int i = 0; i < kFmaCount; ++i) {
    h_a[i] = __half2float(h_left[i]);
    h_b[i] = __half2float(h_right[i]);
    h_c[i] = static_cast<float>((i * 5 % 11) - 5) / 32.0f;
  }
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_ptx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, sizeof(h_a)));
  CUDA_CHECK(cudaMalloc(&d_b, sizeof(h_b)));
  CUDA_CHECK(cudaMalloc(&d_c, sizeof(h_c)));
  CUDA_CHECK(cudaMalloc(&d_ptx, sizeof(h_ptx)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_c, h_c, sizeof(h_c), cudaMemcpyHostToDevice));
  inline_ptx_fma<<<1, 128>>>(d_a, d_b, d_c, d_ptx);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_ptx, d_ptx, sizeof(h_ptx), cudaMemcpyDeviceToHost));
  double ptx_max_abs = 0.0;
  for (int i = 0; i < kFmaCount; ++i) {
    const double expected = std::fma(static_cast<double>(h_a[i]),
                                     static_cast<double>(h_b[i]),
                                     static_cast<double>(h_c[i]));
    ptx_max_abs = std::max(ptx_max_abs, std::abs(h_ptx[i] - expected));
    if (!close(h_ptx[i], expected, 1e-7, 1e-7)) {
      std::cerr << "inline PTX FMA mismatch at element " << i << '\n';
      return 4;
    }
  }

  std::vector<float> h_result(kCount, 0.0f);
  half *d_left = nullptr, *d_right = nullptr;
  float* d_result = nullptr;
  CUDA_CHECK(cudaMalloc(&d_left, sizeof(half) * kCount));
  CUDA_CHECK(cudaMalloc(&d_right, sizeof(half) * kCount));
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(float) * kCount));
  CUDA_CHECK(cudaMemcpy(d_left, h_left.data(), sizeof(half) * kCount,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_right, h_right.data(), sizeof(half) * kCount,
                        cudaMemcpyHostToDevice));
  wmma_16x16x16<<<1, 32>>>(d_left, d_right, d_result);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, sizeof(float) * kCount,
                        cudaMemcpyDeviceToHost));

  double wmma_max_abs = 0.0;
  for (int row = 0; row < kSide; ++row) {
    for (int col = 0; col < kSide; ++col) {
      double expected = 0.0;
      for (int k = 0; k < kSide; ++k) {
        expected += static_cast<double>(__half2float(h_left[row * kSide + k])) *
                    static_cast<double>(__half2float(h_right[k * kSide + col]));
      }
      const float actual = h_result[row * kSide + col];
      wmma_max_abs = std::max(wmma_max_abs, std::abs(actual - expected));
      if (!close(actual, expected, 2e-3, 2e-3)) {
        std::cerr << "WMMA FP16-input/FP32-accumulation mismatch at ["
                  << row << ',' << col << "]\n";
        return 5;
      }
    }
  }
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_ptx));
  CUDA_CHECK(cudaFree(d_left));
  CUDA_CHECK(cudaFree(d_right));
  CUDA_CHECK(cudaFree(d_result));

  std::cout << std::setprecision(9)
            << "{\"status\":\"passed\",\"gpu\":\"" << properties.name
            << "\",\"gpu_uuid\":\"";
  for (const auto byte : properties.uuid.bytes) std::cout << std::hex << std::setw(2)
                                                           << std::setfill('0')
                                                           << (static_cast<unsigned>(byte) & 0xffu);
  std::cout << std::dec << "\",\"compute_capability\":\"" << properties.major
            << '.' << properties.minor << "\",\"checks\":["
            << "{\"name\":\"inline-ptx-fma-rn-f32\",\"max_abs_error\":"
            << ptx_max_abs << "},{\"name\":\"wmma-f16-input-f32-accum\",\"m\":16,\"n\":16,\"k\":16,\"max_abs_error\":"
            << wmma_max_abs << "}]}\n";
  return 0;
}

int timing_smoke(const std::string& input_path) {
  std::vector<half> left, right;
  if (!load_half_inputs(input_path, left, right)) return 2;
  half *d_left = nullptr, *d_right = nullptr;
  float* d_result = nullptr;
  CUDA_CHECK(cudaMalloc(&d_left, sizeof(half) * kCount));
  CUDA_CHECK(cudaMalloc(&d_right, sizeof(half) * kCount));
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(float) * kCount));
  CUDA_CHECK(cudaMemcpy(d_left, left.data(), sizeof(half) * kCount,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_right, right.data(), sizeof(half) * kCount,
                        cudaMemcpyHostToDevice));
  for (int i = 0; i < 20; ++i) wmma_16x16x16<<<1, 32>>>(d_left, d_right, d_result);
  CUDA_CHECK(cudaDeviceSynchronize());
  constexpr int kLaunches = 200;
  constexpr int kSamples = 9;
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> microseconds;
  microseconds.reserve(kSamples);
  for (int sample = 0; sample < kSamples; ++sample) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kLaunches; ++i)
      wmma_16x16x16<<<1, 32>>>(d_left, d_right, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    microseconds.push_back(static_cast<float>(elapsed_ms * 1000.0 / kLaunches));
  }
  std::sort(microseconds.begin(), microseconds.end());
  std::cout << std::setprecision(9)
            << "{\"kernel\":\"wmma_16x16x16\",\"measurement\":\"unprofiled_cuda_event_per_launch_us\",\"launches_per_sample\":"
            << kLaunches << ",\"samples_us\":[";
  for (size_t i = 0; i < microseconds.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << microseconds[i];
  }
  std::cout << "]}\n";
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_left));
  CUDA_CHECK(cudaFree(d_right));
  CUDA_CHECK(cudaFree(d_result));
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "--self-test") == 0 &&
      std::strcmp(argv[2], "--input") == 0) return self_test(argv[3]);
  if (argc == 4 && std::strcmp(argv[1], "--time-smoke") == 0 &&
      std::strcmp(argv[2], "--input") == 0) return timing_smoke(argv[3]);
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    std::cout << "usage: volta_smoke --self-test --input <fp16le-matrices> | "
                 "--time-smoke --input <fp16le-matrices>\n";
    return 0;
  }
  std::cerr << "usage: volta_smoke --self-test --input <fp16le-matrices> | "
               "--time-smoke --input <fp16le-matrices>\n";
  return 64;
}
