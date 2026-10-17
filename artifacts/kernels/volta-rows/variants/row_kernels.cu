// SM70 row kernels for KWV-05. This file is intentionally limited to
// softmax and RMSNorm; it is not a generic reduction library.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include <cmath>

namespace kwv_rows {

constexpr unsigned kFullWarp = 0xffffffffu;
constexpr int kWarp = 32;
constexpr int kCtaThreads = 128;

struct Add {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a + b;
  }
};
struct Max {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a > b ? a : b;
  }
};

// All 32 lanes in the calling warp participate. The kernels below arrange
// for a whole warp/block to take each reduction path uniformly.
template <class BinaryOp>
__device__ __forceinline__ float warp_reduce(float value, BinaryOp op) {
#pragma unroll
  for (int delta = 16; delta; delta >>= 1) {
    value = op(value, __shfl_down_sync(kFullWarp, value, delta));
  }
  return __shfl_sync(kFullWarp, value, 0);
}

template <class BinaryOp>
__device__ __forceinline__ float cta_reduce(float value, float identity,
                                             BinaryOp op, float* shared) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarp - 1);
  const int warp = tid / kWarp;
  value = warp_reduce(value, op);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < (kCtaThreads / kWarp) ? shared[lane] : identity;
    value = warp_reduce(value, op);
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  value = shared[0];
  // Prevent the next reduction's partial stores from racing a lagging reader.
  __syncthreads();
  return value;
}

// One warp owns one row. Arbitrary finite row widths and element masks are
// supported. Masked columns are excluded and written as zero; an all-masked
// row is written as zero without evaluating -inf - -inf.
__global__ void softmax_warp(const float* x, const uint8_t* mask, float* y,
                             int rows, int cols, int x_row_stride,
                             int mask_row_stride, int y_row_stride) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp_in_cta = threadIdx.x / kWarp;
  const int warps_per_cta = blockDim.x / kWarp;
  const int row = blockIdx.x * warps_per_cta + warp_in_cta;
  if (row >= rows) return;  // This whole warp owns the same out-of-range row.

  const float* xr = x + row * x_row_stride;
  float* yr = y + row * y_row_stride;
  float local_max = -CUDART_INF_F;
  for (int col = lane; col < cols; col += kWarp) {
    const bool keep = mask == nullptr || mask[row * mask_row_stride + col] != 0;
    if (keep) local_max = fmaxf(local_max, xr[col]);
  }
  const float row_max = warp_reduce(local_max, Max{});
  const bool all_masked = row_max == -CUDART_INF_F;
  float local_sum = 0.0f;
  if (!all_masked) {
    for (int col = lane; col < cols; col += kWarp) {
      const bool keep = mask == nullptr || mask[row * mask_row_stride + col] != 0;
      if (keep) local_sum += expf(xr[col] - row_max);
    }
  }
  const float denominator = warp_reduce(local_sum, Add{});
  for (int col = lane; col < cols; col += kWarp) {
    const bool keep = mask == nullptr || mask[row * mask_row_stride + col] != 0;
    yr[col] = (keep && !all_masked) ? expf(xr[col] - row_max) / denominator : 0.0f;
  }
}

// One CTA owns one row. Four warps cooperate through four shared FP32 partials.
// Full blocks execute every shuffle and barrier; tails contribute identities.
__global__ void softmax_cta(const float* x, const uint8_t* mask, float* y,
                            int rows, int cols, int x_row_stride,
                            int mask_row_stride, int y_row_stride) {
  __shared__ float partial[kCtaThreads / kWarp];
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + row * x_row_stride;
  float* yr = y + row * y_row_stride;
  float local_max = -CUDART_INF_F;
  for (int col = tid; col < cols; col += kCtaThreads) {
    const bool keep = mask == nullptr || mask[row * mask_row_stride + col] != 0;
    if (keep) local_max = fmaxf(local_max, xr[col]);
  }
  const float row_max = cta_reduce(local_max, -CUDART_INF_F, Max{}, partial);
  const bool all_masked = row_max == -CUDART_INF_F;
  float local_sum = 0.0f;
  if (!all_masked) {
    for (int col = tid; col < cols; col += kCtaThreads) {
      const bool keep = mask == nullptr || mask[row * mask_row_stride + col] != 0;
      if (keep) local_sum += expf(xr[col] - row_max);
    }
  }
  const float denominator = cta_reduce(local_sum, 0.0f, Add{}, partial);
  for (int col = tid; col < cols; col += kCtaThreads) {
    const bool keep = mask == nullptr || mask[row * mask_row_stride + col] != 0;
    yr[col] = (keep && !all_masked) ? expf(xr[col] - row_max) / denominator : 0.0f;
  }
}

// Plain RMSNorm: mean square and scaling are FP32; epsilon is added after the
// mean. Input, gamma, and output are FP32. There is no residual or bias here.
__global__ void rmsnorm_warp(const float* x, const float* gamma, float* y,
                             int rows, int cols, int x_row_stride,
                             int y_row_stride, float epsilon) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp_in_cta = threadIdx.x / kWarp;
  const int warps_per_cta = blockDim.x / kWarp;
  const int row = blockIdx.x * warps_per_cta + warp_in_cta;
  if (row >= rows) return;
  const float* xr = x + row * x_row_stride;
  float* yr = y + row * y_row_stride;
  float local_sumsq = 0.0f;
  for (int col = lane; col < cols; col += kWarp) {
    const float value = xr[col];
    local_sumsq += value * value;
  }
  const float sumsq = warp_reduce(local_sumsq, Add{});
  const float inverse_rms = rsqrtf(epsilon + sumsq / static_cast<float>(cols));
  for (int col = lane; col < cols; col += kWarp) {
    yr[col] = gamma[col] * xr[col] * inverse_rms;
  }
}

__global__ void rmsnorm_cta(const float* x, const float* gamma, float* y,
                            int rows, int cols, int x_row_stride,
                            int y_row_stride, float epsilon) {
  __shared__ float partial[kCtaThreads / kWarp];
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + row * x_row_stride;
  float* yr = y + row * y_row_stride;
  float local_sumsq = 0.0f;
  for (int col = tid; col < cols; col += kCtaThreads) {
    const float value = xr[col];
    local_sumsq += value * value;
  }
  const float sumsq = cta_reduce(local_sumsq, 0.0f, Add{}, partial);
  const float inverse_rms = rsqrtf(epsilon + sumsq / static_cast<float>(cols));
  for (int col = tid; col < cols; col += kCtaThreads) {
    yr[col] = gamma[col] * xr[col] * inverse_rms;
  }
}

// Aligned vector candidate. The caller must dispatch here only when x/y row
// starts and gamma are 16-byte aligned (cudaMalloc satisfies gamma) and row
// strides preserve 16-byte alignment. A scalar tail handles cols % 4 != 0.
__global__ void rmsnorm_cta_vec4(const float* x, const float* gamma, float* y,
                                 int rows, int cols, int x_row_stride,
                                 int y_row_stride, float epsilon) {
  __shared__ float partial[kCtaThreads / kWarp];
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* xr = x + row * x_row_stride;
  float* yr = y + row * y_row_stride;
  const int vector_cols = (cols / 4) * 4;
  const float4* xv = reinterpret_cast<const float4*>(xr);
  const float4* gv = reinterpret_cast<const float4*>(gamma);
  float local_sumsq = 0.0f;
  for (int base = tid * 4; base < vector_cols; base += kCtaThreads * 4) {
    const float4 v = xv[base / 4];
    local_sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  for (int col = vector_cols + tid; col < cols; col += kCtaThreads) {
    const float value = xr[col];
    local_sumsq += value * value;
  }
  const float sumsq = cta_reduce(local_sumsq, 0.0f, Add{}, partial);
  const float inverse_rms = rsqrtf(epsilon + sumsq / static_cast<float>(cols));
  float4* yv = reinterpret_cast<float4*>(yr);
  for (int base = tid * 4; base < vector_cols; base += kCtaThreads * 4) {
    const float4 v = xv[base / 4];
    const float4 g = gv[base / 4];
    yv[base / 4] = make_float4(g.x * v.x * inverse_rms,
                               g.y * v.y * inverse_rms,
                               g.z * v.z * inverse_rms,
                               g.w * v.w * inverse_rms);
  }
  for (int col = vector_cols + tid; col < cols; col += kCtaThreads) {
    yr[col] = gamma[col] * xr[col] * inverse_rms;
  }
}

enum class RmsnormCtaPath { kNoLaunch, kScalar, kVec4 };

// Host dispatch checks the actual device addresses after any leading offset
// has been applied. It also validates logical extents before selecting the
// vector path. The vector kernel handles cols % 4 with a scalar tail, so it
// never reads or writes beyond the logical row width.
cudaError_t launch_rmsnorm_cta(const float* x, const float* gamma, float* y,
                               int rows, int cols, int x_row_stride,
                               int y_row_stride, float epsilon,
                               bool request_vec4, cudaStream_t stream,
                               RmsnormCtaPath* selected_path) {
  if (selected_path) *selected_path = RmsnormCtaPath::kNoLaunch;
  if (rows < 0 || cols < 1 || x_row_stride < cols || y_row_stride < cols ||
      !std::isfinite(epsilon) || epsilon <= 0.0f) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) return cudaSuccess;
  if (x == nullptr || gamma == nullptr || y == nullptr) return cudaErrorInvalidValue;

  const bool aligned16 =
      (reinterpret_cast<uintptr_t>(x) & 15u) == 0 &&
      (reinterpret_cast<uintptr_t>(gamma) & 15u) == 0 &&
      (reinterpret_cast<uintptr_t>(y) & 15u) == 0;
  const bool rows_preserve_alignment = (x_row_stride % 4 == 0) &&
                                      (y_row_stride % 4 == 0);
  if (request_vec4 && cols >= 4 && aligned16 && rows_preserve_alignment) {
    rmsnorm_cta_vec4<<<rows, kCtaThreads, 0, stream>>>(
        x, gamma, y, rows, cols, x_row_stride, y_row_stride, epsilon);
    if (selected_path) *selected_path = RmsnormCtaPath::kVec4;
  } else {
    rmsnorm_cta<<<rows, kCtaThreads, 0, stream>>>(
        x, gamma, y, rows, cols, x_row_stride, y_row_stride, epsilon);
    if (selected_path) *selected_path = RmsnormCtaPath::kScalar;
  }
  return cudaGetLastError();
}

// The native runner skips launch for rows == 0, groups four warp-owned rows
// per 128-thread softmax_warp CTA, and assigns one 128-thread CTA per row for
// cooperative softmax and CTA RMSNorm. RMSNorm candidate dispatch receives
// already-offset pointers so the alignment guard checks the actual addresses.
// Allocation, represented-input parsing, output checks, and timing live in
// row_runner.cu rather than this operator source.

}  // namespace kwv_rows
