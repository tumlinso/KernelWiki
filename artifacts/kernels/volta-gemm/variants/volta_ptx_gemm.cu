#include "volta_ptx_gemm.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace kwv_gemm {
namespace {

__device__ __forceinline__ unsigned pack_half2(__half low, __half high) {
  return static_cast<unsigned>(__half_as_ushort(low)) |
         (static_cast<unsigned>(__half_as_ushort(high)) << 16);
}

template <int TM, int TN, int TK>
__device__ __forceinline__ void load_global_stage(
    const std::uint16_t* a_bits, const std::uint16_t* b_bits,
    __half* shared_a, __half* shared_b, int tid, int threads,
    int block_m, int block_n, int m, int n, int k, int k_tile) {
  for (int index = tid; index < TM * TK; index += threads) {
    const int local_row = index / TK;
    const int local_k = index % TK;
    const int row = block_m + local_row;
    const int kk = k_tile * TK + local_k;
    const std::uint16_t bits = (row < m && kk < k) ? a_bits[row * k + kk] : 0u;
    shared_a[index] = __ushort_as_half(bits);
  }
  for (int index = tid; index < TK * TN; index += threads) {
    const int local_k = index / TN;
    const int local_col = index % TN;
    const int kk = k_tile * TK + local_k;
    const int col = block_n + local_col;
    const std::uint16_t bits = (kk < k && col < n) ? b_bits[kk * n + col] : 0u;
    shared_b[index] = __ushort_as_half(bits);
  }
}

template <int TM, int TN, int TK, int Stages, bool FusedBiasRelu>
__global__ void ptx_gemm_kernel(const std::uint16_t* a_bits,
                                const std::uint16_t* b_bits,
                                const float* bias, float* output,
                                int m, int n, int k) {
  static_assert(TM % 16 == 0 && TN % 16 == 0 && TK % 4 == 0,
                "tiles must align with E09 warp and m8n8k4 instruction tiles");
  static_assert(Stages == 1 || Stages == 2, "only measured one/two-stage variants are in scope");
  constexpr int WarpsM = TM / 16;
  constexpr int WarpsN = TN / 16;
  constexpr int WarpCount = WarpsM * WarpsN;
  constexpr int Threads = WarpCount * 32;
  constexpr int APerThread = (TM * TK + Threads - 1) / Threads;
  constexpr int BPerThread = (TK * TN + Threads - 1) / Threads;

  __shared__ __half shared_a[Stages][TM * TK];
  __shared__ __half shared_b[Stages][TK * TN];

  const int tid = threadIdx.x;
  const int block_m = blockIdx.y * TM;
  const int block_n = blockIdx.x * TN;
  const int k_tiles = (k + TK - 1) / TK;

  if constexpr (Stages == 2) {
    load_global_stage<TM, TN, TK>(a_bits, b_bits, shared_a[0], shared_b[0],
                                   tid, Threads, block_m, block_n, m, n, k, 0);
    __syncthreads();
  }

  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp / WarpsN;
  const int warp_n = warp % WarpsN;
  const int warp_m_origin = block_m + warp_m * 16;
  const int warp_n_origin = block_n + warp_n * 16;
  const int op = (lane >> 2) & 3;
  const int op_m = (op >> 1) * 8;
  const int op_n = (op & 1) * 8;
  const int a_row_local = op_m + (lane & 3) + (lane >= 16 ? 4 : 0);
  const int b_col_local = op_n + (lane & 3) + (lane >= 16 ? 4 : 0);
  // Each warp owns a distinct 16x16 quadrant of the CTA tile.  Indexing only
  // the per-instruction 0..15 coordinates would make every warp reread the
  // same top-left shared-memory tile while storing to a different output tile.
  const int a_row_shared = warp_m * 16 + a_row_local;
  const int b_col_shared = warp_n * 16 + b_col_local;

  float accum[8]{};
  int current_stage = 0;
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    std::uint16_t prefetched_a[APerThread];
    std::uint16_t prefetched_b[BPerThread];
    const bool has_next = k_tile + 1 < k_tiles;

    if constexpr (Stages == 1) {
      load_global_stage<TM, TN, TK>(a_bits, b_bits, shared_a[0], shared_b[0],
                                     tid, Threads, block_m, block_n, m, n, k, k_tile);
      __syncthreads();
    } else {
      // Ordinary global loads into bounded per-thread registers are issued
      // before the current tile's independent MMA instructions. These are
      // synchronous software prefetches, not async-copy instructions.
      if (has_next) {
        #pragma unroll
        for (int slot = 0; slot < APerThread; ++slot) {
          const int index = tid + slot * Threads;
          std::uint16_t bits = 0;
          if (index < TM * TK) {
            const int row = block_m + index / TK;
            const int kk = (k_tile + 1) * TK + index % TK;
            if (row < m && kk < k) bits = a_bits[row * k + kk];
          }
          prefetched_a[slot] = bits;
        }
        #pragma unroll
        for (int slot = 0; slot < BPerThread; ++slot) {
          const int index = tid + slot * Threads;
          std::uint16_t bits = 0;
          if (index < TK * TN) {
            const int kk = (k_tile + 1) * TK + index / TN;
            const int col = block_n + index % TN;
            if (kk < k && col < n) bits = b_bits[kk * n + col];
          }
          prefetched_b[slot] = bits;
        }
      }
    }

    for (int local_k = 0; local_k < TK; local_k += 4) {
      const unsigned a0 = pack_half2(shared_a[current_stage][a_row_shared * TK + local_k + 0],
                                     shared_a[current_stage][a_row_shared * TK + local_k + 1]);
      const unsigned a1 = pack_half2(shared_a[current_stage][a_row_shared * TK + local_k + 2],
                                     shared_a[current_stage][a_row_shared * TK + local_k + 3]);
      const unsigned b0 = pack_half2(shared_b[current_stage][(local_k + 0) * TN + b_col_shared],
                                     shared_b[current_stage][(local_k + 1) * TN + b_col_shared]);
      const unsigned b1 = pack_half2(shared_b[current_stage][(local_k + 2) * TN + b_col_shared],
                                     shared_b[current_stage][(local_k + 3) * TN + b_col_shared]);
      asm volatile(
          "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
          "{%0,%1,%2,%3,%4,%5,%6,%7};"
          : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]),
            "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7])
          : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    }

    __syncthreads();  // Current shared tile is dead for every warp.
    if constexpr (Stages == 2) {
      if (has_next) {
        const int next_stage = 1 - current_stage;
        #pragma unroll
        for (int slot = 0; slot < APerThread; ++slot) {
          const int index = tid + slot * Threads;
          if (index < TM * TK) shared_a[next_stage][index] = __ushort_as_half(prefetched_a[slot]);
        }
        #pragma unroll
        for (int slot = 0; slot < BPerThread; ++slot) {
          const int index = tid + slot * Threads;
          if (index < TK * TN) shared_b[next_stage][index] = __ushort_as_half(prefetched_b[slot]);
        }
        __syncthreads();
        current_stage = next_stage;
      }
    }
  }

  #pragma unroll
  for (int slot = 0; slot < 8; ++slot) {
    const int row_in_op = (lane & 1) + (slot & 2) + (lane >= 16 ? 4 : 0);
    const int col_in_op = (slot & 4) + (lane & 2) + (slot & 1);
    const int row = warp_m_origin + op_m + row_in_op;
    const int col = warp_n_origin + op_n + col_in_op;
    if (row < m && col < n) {
      float value = accum[slot];
      if constexpr (FusedBiasRelu) {
        value += bias[col];
        value = value > 0.0f ? value : 0.0f;
      }
      output[row * n + col] = value;
    }
  }
}

template <int TM, int TN, int TK, int Stages, bool Fused>
cudaError_t launch(const std::uint16_t* a, const std::uint16_t* b,
                   const float* bias, float* output,
                   int m, int n, int k, cudaStream_t stream) {
  constexpr int threads = (TM / 16) * (TN / 16) * 32;
  const dim3 grid((n + TN - 1) / TN, (m + TM - 1) / TM);
  ptx_gemm_kernel<TM, TN, TK, Stages, Fused><<<grid, threads, 0, stream>>>(
      a, b, bias, output, m, n, k);
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_ptx_gemm(const std::uint16_t* a_bits,
                            const std::uint16_t* b_bits,
                            const float* bias, float* output,
                            int m, int n, int k, GemmTile tile,
                            int shared_stages, bool bias_relu,
                            cudaStream_t stream) {
  if ((shared_stages != 1 && shared_stages != 2) || m <= 0 || n <= 0 || k <= 0)
    return cudaErrorInvalidValue;
  if (!a_bits || !b_bits || !output || (bias_relu && !bias)) return cudaErrorInvalidValue;
  switch (tile) {
    case GemmTile::Large64:
      if (shared_stages == 1)
        return bias_relu ? launch<64, 64, 32, 1, true>(a_bits, b_bits, bias, output, m, n, k, stream)
                         : launch<64, 64, 32, 1, false>(a_bits, b_bits, bias, output, m, n, k, stream);
      return bias_relu ? launch<64, 64, 32, 2, true>(a_bits, b_bits, bias, output, m, n, k, stream)
                       : launch<64, 64, 32, 2, false>(a_bits, b_bits, bias, output, m, n, k, stream);
    case GemmTile::SmallM16x32:
      if (m != 16 || n != 4096 || k != 4096) return cudaErrorInvalidValue;
      if (shared_stages == 1)
        return bias_relu ? launch<16, 32, 32, 1, true>(a_bits, b_bits, bias, output, m, n, k, stream)
                         : launch<16, 32, 32, 1, false>(a_bits, b_bits, bias, output, m, n, k, stream);
      return bias_relu ? launch<16, 32, 32, 2, true>(a_bits, b_bits, bias, output, m, n, k, stream)
                       : launch<16, 32, 32, 2, false>(a_bits, b_bits, bias, output, m, n, k, stream);
    case GemmTile::Ragged32x64:
      if (m != 127 || n != 513 || k != 769) return cudaErrorInvalidValue;
      if (shared_stages == 1)
        return bias_relu ? launch<32, 64, 32, 1, true>(a_bits, b_bits, bias, output, m, n, k, stream)
                         : launch<32, 64, 32, 1, false>(a_bits, b_bits, bias, output, m, n, k, stream);
      return bias_relu ? launch<32, 64, 32, 2, true>(a_bits, b_bits, bias, output, m, n, k, stream)
                       : launch<32, 64, 32, 2, false>(a_bits, b_bits, bias, output, m, n, k, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace kwv_gemm
