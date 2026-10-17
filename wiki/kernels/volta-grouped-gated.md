---
id: kernel-volta-grouped-gated
title: Volta Grouped and Gated FP16 Projections
type: kernel
architectures: [sm70]
tags: [grouped-gemm, gated-dual-gemm, gemm, fused-kernel, kernel-fusion, tile-scheduling]
confidence: experimental
reproducibility: benchmarked
kernel_types: [grouped-gemm, gated-dual-gemm, fused-kernel]
languages: [cuda-cpp]
related: [kernel-volta-gemm, kernel-gated-dual-gemm, technique-volta-grouping-and-decoding]
sources: [blog-volta-grouped-lowbit-measurements]
performance_claims:
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 represented inputs/weights; FP32 projection and epilogue
    shape: 128 assignments, 8 experts, K=512, N=256, balanced groups; paired SIMT
    metric: CUDA-event median latency (us)
    value: 69.632
    source_id: blog-volta-grouped-lowbit-measurements
    source_locator: "Native results > Grouped/gated results > weighted balanced > paired event median"
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 represented inputs/weights; FP32 projection and epilogue
    shape: 128 assignments, 8 experts, K=512, N=256, skewed groups; split cuBLAS
    metric: CUDA-event median latency (us)
    value: 45.056
    source_id: blog-volta-grouped-lowbit-measurements
    source_locator: "Native results > Grouped/gated results > weighted skewed > split event median"
artifact_dir: artifacts/kernels/volta-grouped-lowbit/variants
---

# Operator contract

For each assignment `(row,expert,pos)`, compute
`U = X[row] @ W_up[expert]`, `G = X[row] @ W_gate[expert]`, and
`Y[pos] = silu(G) * U`. Inputs and expert weights are represented FP16.
Projection accumulators and materialized gate/up values are FP32; the SiLU,
product, and output are also FP32. The assignment identity and output position
are both explicit.

Rows may be assigned to the same expert more than once, including repeated
`(row,expert)` pairs. Such records remain distinct outputs. Empty experts are
retained in the stable expert offset table. The tested skewed case has counts
`[1,0,0,0,0,0,0,127]`: one singleton group, one heavy group, and six empty
groups.

The paired SIMT kernel computes both FP32 projections from the same staged row,
then applies the gated epilogue:

```cuda
for (int kk = 0; kk < k; ++kk) {
  const float x = __half2float(x_row[kk]);
  up = fmaf(x, wu, up);
  gate = fmaf(x, wg, gate);
}
const float silu = gate / (1.0f + expf(-gate));
output[output_position * n + col] = silu * up;
```

# Compared mappings

The library baseline groups assignment indices on the host, gathers all
assigned X rows, calls cuBLAS for separate up and gate projections on every
nonempty expert, then applies the FP32 SiLU/product and scatters results to
their original output positions. Its full device path includes gather, every
projection, intermediate reads/writes, epilogue, and scatter.

The paired candidate is a bounded small-M SIMT path. One CTA/output-column
tile stages a single assigned X row once, computes both projections into
paired FP32 accumulators, then writes the gated output to the explicit
position. It does not claim native Tensor Core fusion and is not a second
general GEMM engine. Results are shape- and distribution-specific; the
measured tables and the full host/device timing split are in the
[measurement record](../../sources/blogs/volta-grouped-lowbit-measurements.md).

## Measured boundary

On this V100, the paired path won the measured tiny, moderate, and weighted
balanced cases. For the weighted balanced fixture, median device-event time
was 190.464 µs for the split cuBLAS route and 69.632 µs for the paired SIMT
route; synchronized host medians were 198.305 and 76.667 µs. The weighted
skewed fixture reversed the result: 45.056/52.749 µs for split device/host
medians and 66.560/73.920 µs for paired. The balanced fixture has eight
nonempty expert groups; the skewed one has two, so the split baseline issues
different numbers of library projection calls and sees different row sizes.
This comparison supports a workload-dependent dispatch boundary, not an
isolated fusion-only causal claim, because the routes use different GEMM
engines.

All six suite cases passed in the independent native checks for both routes.
The event scope includes gather, both projections for every nonempty expert,
the FP32 SiLU/product, and scatter. Stable assignment grouping and metadata
preparation are host setup, outside the CUDA event. Synchronized host time
includes submission through completion. Input/oracle reads, validation,
allocation, and H2D synchronization are reported separately; the assignment
metadata time is a subcomponent of setup, not an extra term. The report's
setup-plus-warm-use host value is derived arithmetic and is not a measured
cold application request. The complete timing table, case sizes, setup values,
and limitations are in the linked measurement record and
[`benchmark_runs/volta-grouped-lowbit/summary.json`](../../benchmark_runs/volta-grouped-lowbit/summary.json).

# Numerical admission and evidence

The predeclared per-output FP32 admission policy derives a `gamma_K` allowance
for both projections, propagates it through the SiLU derivative bound, and
adds the fixed FP32 activation/product allowance. Every native output is
compared with an independent FP64 oracle generated from the exact FP16 input
bits. The numerical contract and suite membership are in
[`benchmark_runs/volta-grouped-lowbit/cases/fixture_bundle/`](../../benchmark_runs/volta-grouped-lowbit/README.md).
The source implementation and native role evidence are linked from
[`variants/`](../../artifacts/kernels/volta-grouped-lowbit/variants/README.md).
