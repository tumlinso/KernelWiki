---
id: technique-volta-fragment-layout
title: Volta explicit MMA fragment ownership and pointwise consumers
type: technique
architectures: [sm70]
tags: [mma-sm70, gemm, kernel-fusion]
confidence: experimental
reproducibility: benchmarked
prerequisites: [hw-volta-mma]
related: [hw-volta-mma, kernel-volta-gemm, technique-volta-software-staging]
sources: [blog-volta-atlas-campaign, blog-volta-gemm-measurements]
artifact_dir: artifacts/kernels/volta-gemm/variants
---

# Explicit PTX ownership

The explicit FP16-input, FP32-accumulator `mma.sync.aligned.m8n8k4` path is a
warp collective. Its accumulator lane/slot map must be kept distinct from raw
WMMA storage; this experiment does not infer WMMA ownership. The E09 atlas
coordinate map was reused, and the newly built runner passed a short exact
one-hot compatibility check over the 256 expected coordinates
(`benchmark_runs/volta-gemm/evidence/`, controller evidence
`32658912-e7c4-41a6-866e-4010c07682e6`, runner SHA-256
`fd6989c76a9d3925236771e4c7e8a9f70cda5fa2820420d1d1623e8408330ac7`). That
K=4 smoke does not exercise the stage-2 K-loop; full-K correctness cases do.

The implementation lifts the per-instruction 16x16 coordinates into the
selected CTA tile by including each warp's tile origin:

```cpp
const int warp_m = warp / WarpsN;
const int warp_n = warp % WarpsN;
const int a_row_shared = warp_m * 16 + a_row_local;
const int b_col_shared = warp_n * 16 + b_col_local;
```

Ignoring either warp offset makes multiple warps consume the same top-left
shared-memory fragment while writing to different output coordinates. The
PTX output map and shared reads are in
`volta_ptx_gemm.cu`; the exact-source bundle includes the short E09 smoke.

# Pointwise epilogue ownership

For the measured bias/ReLU consumer, the lane and accumulator slot that own
`C[m,n]` also apply `C[m,n] += bias[n]` and ReLU before storing. Therefore
`P(x)=Q(x)` for every output coordinate: the required ownership transform is
identity. There are no cross-lane words, shuffles, shared-memory conversions,
or second MMA to remove in this case.

| Case | cuBLAS GEMM + separate bias/ReLU (us) | fused PTX stage 2 (us) |
|---|---:|---:|
| `large_dense` | `77.824 ± 2.048` | `232.448 ± 1.024` |
| `small_m` | `91.136 ± 1.024` | `245.760 ± 1.024` |
| `ragged` | `53.248 ± 1.024` | `47.104 ± 0.000` |

These are full device-operator event intervals. The fused route avoids a
materialized intermediate and separate epilogue launch, but it also uses a
different GEMM engine. The timings answer a full-dispatch choice and cannot
isolate fusion as the cause of the difference. The ragged PTX/cuBLAS result is
also too close to separate from the measured variability; CUTLASS was faster
on that exact ragged case. See
[`volta-gemm-measurements`](../../sources/blogs/volta-gemm-measurements.md).
