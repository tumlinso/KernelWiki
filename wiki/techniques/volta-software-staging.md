---
id: technique-volta-software-staging
title: Volta GEMM software staging and stage-count tradeoffs
type: technique
architectures: [sm70]
tags: [mma-sm70, gemm, pipeline-stages]
confidence: experimental
reproducibility: benchmarked
prerequisites: [hw-volta-mma]
related: [hw-volta-mma, kernel-volta-gemm, technique-volta-fragment-layout]
sources: [blog-volta-gemm-measurements]
artifact_dir: artifacts/kernels/volta-gemm/variants
---

# One and two shared-memory stages

The candidate holds FP32 accumulators while it walks K in `m8n8k4` steps.
Stage 1 loads a tile to shared memory, synchronizes the CTA, and performs its
MMA steps. Stage 2 preloads the first tile, issues ordinary global loads for
the next tile into per-thread registers while the current shared tile feeds
MMA, then stores the registers into the alternate shared buffer after the
current tile is consumed. CTA barriers protect buffer reuse.

The two-stage path is software staging, not `cp.async`, TMA, or `mbarrier`.
The load-prefetch arrays are visible in
`artifacts/kernels/volta-gemm/variants/volta_ptx_gemm.cu`:

```cpp
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
```

The following barrier and alternate-stage stores in that same source follow
the current tile's MMA loop, so the shared-memory buffer is not reused before
the consumer finishes.

# Measured stage-count result

All timings below use the same PTX math and shape-specific tile. Each value is
the median and MAD of 11 interleaved single-sample CUDA-event invocations on
the same V100. The events cover the complete selected GEMM device operator;
allocation and host transfers are outside the interval.

| Case | PTX tile | Stage 1 (us) | Stage 2 (us) | PTX stage-2 change | Useful/executed products |
|---|---:|---:|---:|---:|---:|
| `large_dense` | `64x64x32` | `250.880 ± 1.024` | `231.424 ± 1.024` | `-7.8%` | 100% |
| `small_m` | `16x32x32` | `561.152 ± 0.000` | `244.736 ± 1.024` | `-56.4%` | 100% |
| `ragged` | `32x64x32` | `61.440 ± 0.000` | `48.128 ± 1.024` | `-21.7%` | 84.94% |

Two stages improve all three measured PTX cases, but cuBLAS is faster on the
dense and small-M workloads. On ragged, CUTLASS is the faster measured
alternative (`34.816 µs`); the PTX/cuBLAS stage-2 difference is inconclusive
at the observed event granularity. A stage-count gain within this simple
candidate does not make it the best overall implementation.

One selected Nsight Compute sample was collected for the stage-2 PTX kernel
on each case. Counter capture is valid, replay timing is not; each report is
partial/mixed with one kernel sample and asks for more data. The profile does
not identify staging as the causal bottleneck. Keep the result at the level
of measured full-operator times and this tested software pipeline.
