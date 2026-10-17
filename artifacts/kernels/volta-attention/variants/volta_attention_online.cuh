#pragma once

// KernelWiki-authored KWV-06 SIMT online attention implementation.
// Launch `volta_attention_online_rows<<<dim3(rows, splits), 128, 0, stream>>>`.
// The block must contain exactly four full warps. Inputs are contiguous
// row-major [B,H,S,D] FP16. D must be in [1,128].

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>

namespace kernelwiki_volta_attention {

constexpr int kWarps = 4;
constexpr int kWarpWidth = 32;
constexpr int kMaxHeadDim = 128;
constexpr unsigned kFullWarpMask = 0xffffffffu;

// `selected_counts == nullptr` means dense attention. Otherwise each row has
// selected_ids[row * max_selected_blocks + slot] block IDs. The host validates
// IDs, counts, and duplicate rejection before launch. `masked` is optional and
// uses 1 for excluded entries in row-major [rows, Sk] layout.
__global__ void online_rows(const half* q, const half* k, const half* v,
                            const uint8_t* masked,
                            const int32_t* selected_ids,
                            const int32_t* selected_counts,
                            int rows, int hq, int hkv, int sq, int sk, int d,
                            int causal, int block_size,
                            int max_selected_blocks, int split_count,
                            float* output, float* row_max,
                            float* row_normalizer, float* partial_m,
                            float* partial_l, float* partial_u) {
  const int row = static_cast<int>(blockIdx.x);
  const int split = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x) & (kWarpWidth - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarpWidth;
  if (blockDim.x != kWarps * kWarpWidth || row >= rows || d <= 0 ||
      d > kMaxHeadDim || hq <= 0 || hkv <= 0 || hq % hkv != 0 ||
      sq <= 0 || sk <= 0 || (causal && sq > sk) || split_count <= 0 ||
      split >= split_count ||
      (selected_counts != nullptr &&
       (selected_ids == nullptr || block_size <= 0 || max_selected_blocks <= 0)) ||
      output == nullptr || row_max == nullptr || row_normalizer == nullptr ||
      (split_count > 1 &&
       (partial_m == nullptr || partial_l == nullptr || partial_u == nullptr))) return;

  // Rows are flattened in [B,H,S] order: sequence is the innermost axis.
  const int query_head = (row / sq) % hq;
  const int query_row = row % sq;
  const int batch = row / (hq * sq);
  const int kv_head = query_head / (hq / hkv);
  const int64_t qbase =
      ((static_cast<int64_t>(batch) * hq + query_head) * sq + query_row) * d;
  const int64_t kv_head_base =
      (static_cast<int64_t>(batch) * hkv + kv_head) * sk * d;
  const int64_t split_lo = (static_cast<int64_t>(sk) * split) / split_count;
  const int64_t split_hi = (static_cast<int64_t>(sk) * (split + 1)) / split_count;
  const float scale = rsqrtf(static_cast<float>(d));
  const int selected_count = selected_counts == nullptr
                                 ? 0
                                 : selected_counts[row];

  float running_m = -INFINITY;
  float running_l = 0.0f;
  float running_u[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#define KWV_CONSUME_KEY(KEY_VALUE)                                                \
  do {                                                                            \
    const int key_index = (KEY_VALUE);                                             \
    if (key_index >= split_lo && key_index < split_hi && key_index < sk &&        \
        !(causal && key_index > (sk - sq + query_row)) &&                         \
        (masked == nullptr || masked[static_cast<int64_t>(row) * sk + key_index] == 0)) { \
      float dot = 0.0f;                                                           \
      const int64_t key_base = kv_head_base + static_cast<int64_t>(key_index) * d; \
      for (int chunk = 0; chunk < 4; ++chunk) {                                   \
        const int feature = lane + chunk * kWarpWidth;                            \
        if (feature < d) {                                                        \
          dot = fmaf(__half2float(q[qbase + feature]),                            \
                     __half2float(k[key_base + feature]), dot);                   \
        }                                                                         \
      }                                                                           \
      for (int offset = kWarpWidth / 2; offset > 0; offset >>= 1) {                \
        dot += __shfl_down_sync(kFullWarpMask, dot, offset);                      \
      }                                                                           \
      dot = __shfl_sync(kFullWarpMask, dot, 0);                                   \
      const float score = dot * scale;                                            \
      const float next_m = fmaxf(running_m, score);                               \
      const float old_scale = running_l == 0.0f ? 0.0f : expf(running_m - next_m);\
      const float new_weight = expf(score - next_m);                              \
      running_l = old_scale * running_l + new_weight;                            \
      for (int chunk = 0; chunk < 4; ++chunk) {                                   \
        const int feature = lane + chunk * kWarpWidth;                            \
        if (feature < d) {                                                        \
          const float value = __half2float(v[key_base + feature]);                \
          running_u[chunk] = old_scale * running_u[chunk] + new_weight * value;   \
        }                                                                         \
      }                                                                           \
      running_m = next_m;                                                         \
    }                                                                             \
  } while (0)

  if (selected_counts == nullptr) {
    for (int64_t key = split_lo + warp; key < split_hi; key += kWarps) {
      KWV_CONSUME_KEY(static_cast<int>(key));
    }
  } else {
    for (int slot = warp; slot < selected_count; slot += kWarps) {
      const int block_id = selected_ids[row * max_selected_blocks + slot];
      const int block_begin = block_id * block_size;
      const int block_end = block_begin + block_size < sk ? block_begin + block_size : sk;
      for (int key = block_begin; key < block_end; ++key) {
        KWV_CONSUME_KEY(key);
      }
    }
  }
#undef KWV_CONSUME_KEY

  __shared__ float warp_m[kWarps];
  __shared__ float warp_l[kWarps];
  __shared__ float warp_u[kWarps][kMaxHeadDim];
  if (lane == 0) {
    warp_m[warp] = running_m;
    warp_l[warp] = running_l;
  }
  for (int chunk = 0; chunk < 4; ++chunk) {
    const int feature = lane + chunk * kWarpWidth;
    if (feature < d) warp_u[warp][feature] = running_u[chunk];
  }
  __syncthreads();

  // Merge the four in-CTA key partitions. Empty partitions are skipped before
  // subtracting maxima, avoiding -inf - -inf. This also supplies the state
  // format used by the cross-CTA split-KV merge below.
  if (warp == 0) {
    for (int chunk = 0; chunk < 4; ++chunk) {
      const int feature = lane + chunk * kWarpWidth;
      if (feature < d) {
        float merged_m = -INFINITY;
        float merged_l = 0.0f;
        float merged_u = 0.0f;
        for (int part = 0; part < kWarps; ++part) {
          const float part_l = warp_l[part];
          if (part_l > 0.0f) {
            const float part_m = warp_m[part];
            const float next_m = fmaxf(merged_m, part_m);
            const float old_scale = merged_l == 0.0f ? 0.0f : expf(merged_m - next_m);
            const float part_scale = expf(part_m - next_m);
            merged_l = old_scale * merged_l + part_scale * part_l;
            merged_u = old_scale * merged_u + part_scale * warp_u[part][feature];
            merged_m = next_m;
          }
        }
        if (split_count == 1) {
          output[static_cast<int64_t>(row) * d + feature] =
              merged_l == 0.0f ? 0.0f : merged_u / merged_l;
          if (lane == 0 && chunk == 0) {
            row_max[row] = merged_m;
            row_normalizer[row] = merged_l;
          }
        } else {
          const int64_t state = static_cast<int64_t>(row) * split_count + split;
          if (lane == 0 && chunk == 0) {
            partial_m[state] = merged_m;
            partial_l[state] = merged_l;
          }
          partial_u[state * d + feature] = merged_u;
        }
      }
    }
  }
}

// One CTA per query row. Partial states were already locally merged over four
// warps; this second level merges the independently scheduled KV partitions.
__global__ void merge_split_kv(const float* partial_m, const float* partial_l,
                               const float* partial_u, int rows, int d,
                               int split_count, float* output,
                               float* row_max, float* row_normalizer) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x) & (kWarpWidth - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarpWidth;
  if (blockDim.x != kWarps * kWarpWidth || row >= rows || d <= 0 ||
      d > kMaxHeadDim || split_count <= 0 || partial_m == nullptr ||
      partial_l == nullptr || partial_u == nullptr || output == nullptr ||
      row_max == nullptr || row_normalizer == nullptr) return;
  // Each row CTA needs only one writer warp. The 32 lanes cover up to 128
  // output features through the four chunks below; letting all four warps run
  // this loop creates duplicate global stores to the same locations.
  if (warp != 0) return;
  for (int chunk = 0; chunk < 4; ++chunk) {
    const int feature = lane + chunk * kWarpWidth;
    if (feature < d) {
      float merged_m = -INFINITY;
      float merged_l = 0.0f;
      float merged_u = 0.0f;
      for (int part = 0; part < split_count; ++part) {
        const int64_t state = static_cast<int64_t>(row) * split_count + part;
        const float part_l = partial_l[state];
        if (part_l > 0.0f) {
          const float part_m = partial_m[state];
          const float next_m = fmaxf(merged_m, part_m);
          const float old_scale = merged_l == 0.0f ? 0.0f : expf(merged_m - next_m);
          const float part_scale = expf(part_m - next_m);
          merged_l = old_scale * merged_l + part_scale * part_l;
          merged_u = old_scale * merged_u + part_scale *
                     partial_u[state * d + feature];
          merged_m = next_m;
        }
      }
      output[static_cast<int64_t>(row) * d + feature] =
          merged_l == 0.0f ? 0.0f : merged_u / merged_l;
      if (lane == 0 && chunk == 0) {
        row_max[row] = merged_m;
        row_normalizer[row] = merged_l;
      }
    }
  }
}

}  // namespace kernelwiki_volta_attention
