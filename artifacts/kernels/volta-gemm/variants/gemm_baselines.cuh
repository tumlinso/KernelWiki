#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cutlass/arch/arch.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace kwv_gemm {

// Pinned CUTLASS 2.11.0 Sm70 reference configuration found in its own
// gemm_f16t_f16t_f32t_volta_tensor_op_f32_sm70 tests: CTA 64x64x32,
// warp 32x32x32, instruction 8x8x4, two stages.  A future active KWV-04
// build must qualify this exact instantiation under CUDA 12.9.
using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2,
    1, 1>;

class CutlassRunner {
 public:
  CutlassRunner(int m, int n, int k, const std::uint16_t* a_bits,
                const std::uint16_t* b_bits, float* c, cudaStream_t stream);
  ~CutlassRunner();
  CutlassRunner(const CutlassRunner&) = delete;
  CutlassRunner& operator=(const CutlassRunner&) = delete;

  // Enqueue only the already-initialized GEMM. Allocation, can_implement and
  // initialize are deliberately outside the measured full-operator region.
  cutlass::Status run();

 private:
  CutlassGemm gemm_;
  void* workspace_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

// Row-major cuBLAS is expressed as C^T = B^T A^T through its column-major
// interface. Both library paths accept identical represented half bits and
// write row-major FP32 C. The caller keeps setup/handle creation outside timing.
cublasStatus_t launch_cublas_gemm(cublasHandle_t handle, int m, int n, int k,
                                  const std::uint16_t* a_bits,
                                  const std::uint16_t* b_bits, float* c);

// The unfused G2 boundary is one extra C read/write kernel. The elementwise
// output map is coordinate identity; this operation does not justify a
// register shuffle or a second-MMA ownership claim.
cudaError_t launch_bias_relu(const float* c, const float* bias, float* d,
                             int m, int n, cudaStream_t stream);

}  // namespace kwv_gemm
