---
id: kernel-volta-row-kernels
title: Volta Stable Softmax and RMSNorm Row Kernels
type: kernel
architectures: [sm70]
tags: [reduction, mma-sm70, vectorized-loads]
confidence: experimental
reproducibility: benchmarked
kernel_types: [reduction]
languages: [cuda-cpp]
related: [technique-volta-memory-fusion, pattern-volta-memory-and-residency]
sources: [blog-volta-rows-measurements]
performance_claims:
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP32 input / FP32 accumulation and output
    shape: 1 x 4096 stable softmax; one warp-row baseline
    metric: CUDA-event median latency (us)
    value: 24.576
    source_id: blog-volta-rows-measurements
    source_locator: matched baseline timing, softmax_wide_single_row, 21 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP32 input / FP32 accumulation and output
    shape: 1 x 4096 stable softmax; one CTA-row candidate
    metric: CUDA-event median latency (us)
    value: 11.264
    source_id: blog-volta-rows-measurements
    source_locator: matched candidate timing, softmax_wide_single_row, 21 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP32 input / FP32 accumulation and output
    shape: 512 x 4096 stable softmax; one warp-row baseline
    metric: CUDA-event median latency (us)
    value: 92.160
    source_id: blog-volta-rows-measurements
    source_locator: matched baseline timing, softmax_wide_streaming, 21 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP32 input / FP32 accumulation and output
    shape: 512 x 4096 stable softmax; one CTA-row candidate
    metric: CUDA-event median latency (us)
    value: 47.104
    source_id: blog-volta-rows-measurements
    source_locator: matched candidate timing, softmax_wide_streaming, 21 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP32 input / FP32 accumulation and output
    shape: 512 x 4096 plain RMSNorm; scalar CTA baseline
    metric: CUDA-event median latency (us)
    value: 44.032
    source_id: blog-volta-rows-measurements
    source_locator: matched baseline timing, rmsnorm_vec4_wide_streaming, 21 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP32 input / FP32 accumulation and output
    shape: 512 x 4096 plain RMSNorm; aligned float4 CTA candidate
    metric: CUDA-event median latency (us)
    value: 36.864
    source_id: blog-volta-rows-measurements
    source_locator: matched candidate timing, rmsnorm_vec4_wide_streaming, 21 samples
artifact_dir: artifacts/kernels/volta-rows/variants
---

# Operators and correctness contract

The artifact implements stable row softmax and plain RMSNorm with FP32 device
accumulation and output on SM70. Softmax subtracts each row's active maximum
before exponentiation. Masked outputs are exactly zero; a row with no active
elements returns all zeros. RMSNorm applies
`gamma * x * rsqrt(epsilon + mean(x*x))`. Widths satisfy `1 <= N <= 8193`;
zero rows are valid and launch no kernel.

The fixed suite has 22 cases for widths 31, 32, 33, 128, 768, 4096, and 8193,
plus masks, tails, independent strides, offsets, all-masked rows, a zero-valued
RMSNorm row, misalignment, and zero-row-count behavior. Its represented FP32
inputs and FP64 reference are immutable artifacts. Every logical output is
checked with the predeclared rule
`abs(actual - reference) <= 2^-30 + 2^-14*abs(reference)`. The two native
implementations each passed 4,278,449 output checks with no failures,
non-finite values, or padding writes.

# Mappings

The cooperative CTA reduction in the retained source combines four warp
partials. Tail lanes start with the supplied identity, and every lane reaches
each shuffle and barrier:

```cuda
const float row_max = cta_reduce(local_max, -CUDART_INF_F, Max{}, partial);
const bool all_masked = row_max == -CUDART_INF_F;
const float denominator = cta_reduce(local_sum, 0.0f, Add{}, partial);
```

This excerpt is from `row_kernels.cu`; it shows the reduction boundary rather
than a complete standalone kernel.

The baseline maps one warp to each softmax row and places four rows in a
128-thread CTA. The candidate assigns a 128-thread CTA to each row. All lanes
in each participating warp execute synchronized shuffles; inactive tail and
masked values contribute the reduction identity. The CTA path reduces through
shared partials and barriers.

For RMSNorm, the baseline uses a scalar one-CTA-per-row kernel. The candidate
uses a `float4` load/store path when the actual input, gamma, and output
addresses and row strides satisfy 16-byte alignment. Complete groups of four
are vectorized and the remainder is scalar. The dispatcher selects scalar CTA
fallback when these conditions do not hold; the fixed misaligned case checks
that route. The check and timing outputs report the route actually selected.

These are explicit reference routes rather than a reusable reduction library.
The published source copy is in `artifacts/kernels/volta-rows/variants/row_kernels.cu`; the native
runner, build identity, fixture contract, and controller records are described
in that directory's README and `PROVENANCE.yaml`.

# Measured choices

On the tested V100, the candidate CTA softmax took `11.264 ± 0.000 µs` for one
4096-column row versus `24.576 ± 0.000 µs` for the four-row warp mapping. At
512 rows by 4096 columns, it took `47.104 ± 1.024 µs` versus
`92.160 ± 1.024 µs`. On the matching streaming RMSNorm case, aligned `float4`
took `36.864 ± 1.024 µs` versus `44.032 ± 1.024 µs` for the scalar baseline.
Each value is a 21-sample unprofiled CUDA-event median with MAD, five warmups,
and two alternating buffers. See the [measurement record](../../sources/blogs/volta-rows-measurements.md)
for exact samples, buffer sizes, evidence, and limits.

This evidence supports those three tested route comparisons only. It does not
establish a general width crossover, vectorization benefit for unsupported
alignment, or the cause of the timing differences. Nsight Compute profiled one
softmax candidate kernel; it cannot explain the RMSNorm result. The current
operators also do not contain residual or bias terms, so there is no measured
fusion comparison.

# Validation status

Both correctness runs used the complete 22-case suite. Candidate memcheck used
the exact candidate correctness-checked ELF and reports zero raw errors, but
the wrapper summary remains partial and nonconclusive. NCU counters are valid
for one candidate streaming-softmax launch; profiler timing is invalid under
replay and the summary requests more data. The complete source, immutable
inputs/reference/oracle, controller outputs, raw profiler files, and binary
hashes are linked from `benchmark_runs/volta-rows/summary.json`.
