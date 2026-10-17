#include "gemm_baselines.cuh"

#include <cuda_fp16.h>

#include <sstream>

namespace kwv_gemm {
namespace {

void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

template <class T>
cutlass::TensorRef<const T, cutlass::layout::RowMajor> const_row_major_ref(
    const T* pointer, int stride) {
  return {pointer, cutlass::layout::RowMajor(stride)};
}

__global__ void bias_relu_kernel(const float* c, const float* bias, float* d,
                                 int count, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float value = c[index] + bias[index % n];
    d[index] = value > 0.0f ? value : 0.0f;
  }
}

}  // namespace

CutlassRunner::CutlassRunner(int m, int n, int k,
                             const std::uint16_t* a_bits,
                             const std::uint16_t* b_bits, float* c,
                             cudaStream_t stream) {
  stream_ = stream;
  static_assert(sizeof(cutlass::half_t) == sizeof(std::uint16_t),
                "represented input storage must be IEEE binary16");
  const cutlass::gemm::GemmCoord problem_size(m, n, k);
  auto a = const_row_major_ref(
      reinterpret_cast<const cutlass::half_t*>(a_bits), k);
  auto b = const_row_major_ref(
      reinterpret_cast<const cutlass::half_t*>(b_bits), n);
  cutlass::TensorRef<float, cutlass::layout::RowMajor> c_ref(
      c, cutlass::layout::RowMajor(n));
  // With beta=0 C and D are the same allocation: the operator is a fresh
  // overwrite, not a read/modify/write GEMM. This equality is intentional.
  CutlassGemm::Arguments arguments{problem_size, a, b, c_ref, c_ref,
                                   {1.0f, 0.0f}, 1};
  const auto supported = gemm_.can_implement(arguments);
  if (supported != cutlass::Status::kSuccess)
    throw std::runtime_error("CUTLASS Sm70 can_implement rejected the case");
  const std::size_t bytes = CutlassGemm::get_workspace_size(arguments);
  if (bytes != 0) require_cuda(cudaMalloc(&workspace_, bytes), "cudaMalloc CUTLASS workspace");
  const auto initialized = gemm_.initialize(arguments, workspace_, stream);
  if (initialized != cutlass::Status::kSuccess)
    throw std::runtime_error("CUTLASS Sm70 initialize failed");
}

CutlassRunner::~CutlassRunner() {
  if (workspace_) cudaFree(workspace_);
}

cutlass::Status CutlassRunner::run() { return gemm_(stream_); }

cublasStatus_t launch_cublas_gemm(cublasHandle_t handle, int m, int n, int k,
                                  const std::uint16_t* a_bits,
                                  const std::uint16_t* b_bits, float* c) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Memory is interpreted column-major as B^T[N,K] * A^T[K,M], so the
  // resulting C^T[N,M] has exactly the same row-major bytes as C[M,N].
  return cublasGemmEx(handle,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      n, m, k,
                      &alpha,
                      b_bits, CUDA_R_16F, n,
                      a_bits, CUDA_R_16F, k,
                      &beta,
                      c, CUDA_R_32F, n,
                      CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

cudaError_t launch_bias_relu(const float* c, const float* bias, float* d,
                             int m, int n, cudaStream_t stream) {
  constexpr int threads = 256;
  const int count = m * n;
  const int blocks = (count + threads - 1) / threads;
  bias_relu_kernel<<<blocks, threads, 0, stream>>>(c, bias, d, count, n);
  return cudaGetLastError();
}

}  // namespace kwv_gemm
