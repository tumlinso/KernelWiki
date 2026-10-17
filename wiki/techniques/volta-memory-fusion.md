---
id: technique-volta-memory-fusion
title: Memory Mapping and Fusion for Volta Row Operators
type: technique
architectures: [sm70]
tags: [vectorized-loads, kernel-fusion, shared-memory-optimization]
confidence: experimental
reproducibility: benchmarked
prerequisites: []
related: [kernel-volta-row-kernels, pattern-volta-memory-and-residency]
sources: [blog-volta-rows-measurements]
blackwell_relevance: Row reductions and memory fusion still require shape- and alignment-specific evidence on Volta; newer architectures need their own measurements.
---

# Keep operator boundaries explicit

The current SM70 comparison covers stable softmax and plain RMSNorm. Neither
operator includes residual addition or bias. There is no valid fused-versus-
unfused full-operator comparison in this result, so it supports no fusion
speedup claim.

A useful fusion experiment must define the complete operator on both sides.
For example, compare a fused RMSNorm residual+bias path against the matching
unfused sequence that materializes the same intermediate, with identical
represented input, masks, epsilon, gamma, and output checks. Include every
kernel in the unfused timing boundary and every fused operation in the fused
boundary. A standalone RMSNorm comparison cannot predict that result.

# Alignment-aware vectorization

The candidate dispatches to its vector kernel only when the actual pointers
and row strides satisfy the alignment contract:

```cuda
if (request_vec4 && cols >= 4 && aligned16 && rows_preserve_alignment) {
  rmsnorm_cta_vec4<<<rows, kCtaThreads, 0, stream>>>(
      x, gamma, y, rows, cols, x_row_stride, y_row_stride, epsilon);
}
```

The full runner includes the scalar fallback and tail; this excerpt is the
guarded dispatch site from `row_kernels.cu`.

For the 512 × 4096 RMSNorm case, the guarded `float4` CTA route measured
`36.864 ± 1.024 µs`; scalar CTA measured `44.032 ± 1.024 µs` on the same V100
contract. The candidate checks actual post-offset pointers for input, gamma,
and output and checks row-stride compatibility before vector access. It
processes complete groups of four and uses a scalar tail. Misaligned data uses
the scalar fallback. This one aligned streaming result does not establish a
universal vector width or explain the measured difference by itself.

When evaluating another shape, keep the operator and work mapping fixed while
testing scalar and vector routes. Verify pointer alignment after offsets,
row-stride alignment, tail bounds, and output padding independently. Preserve a
scalar fallback for unsupported layouts unless the API excludes them.

# Evidence boundary

The selected Nsight Compute capture was one candidate softmax streaming launch,
not the RMSNorm kernel. It was classified as memory-bound from one counter
sample, with 44.01% DRAM and 20.30% SM throughput; its wrapper status was
partial, counter collection valid, replay timing invalid, and more data
requested. This does not identify the reason for either unprofiled timing
result and cannot be used as the RMSNorm profile.

The reported 33,587,200-byte rotated RMSNorm working set exceeds the device's
6,291,456-byte L2 capacity. Capacity does not prove actual HBM traffic. To
evaluate fusion, capture and validate the fused RMSNorm path itself, then
measure the same full unfused path and obtain relevant bytes/counter evidence.
The fixed source and raw records are linked in
[`volta-rows-measurements`](../../sources/blogs/volta-rows-measurements.md).
