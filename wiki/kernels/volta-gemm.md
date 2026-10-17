---
id: kernel-volta-gemm
title: Volta FP16-input, FP32-accumulating GEMM
type: kernel
architectures: [sm70]
tags: [gemm, mma-sm70, tile-scheduling, kernel-fusion]
confidence: experimental
reproducibility: benchmarked
kernel_types: [gemm]
languages: [cuda-cpp]
related: [hw-volta-mma, technique-volta-fragment-layout, technique-volta-software-staging]
sources: [blog-volta-atlas-campaign, blog-volta-gemm-measurements]
performance_claims:
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 input / FP32 accumulation and output
    shape: M=1024,N=1024,K=1024 cuBLAS core
    metric: CUDA-event median latency (us)
    value: 61.440
    source_id: blog-volta-gemm-measurements
    source_locator: primary matched timing table, large_dense cuBLAS core, 11 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 input / FP32 accumulation and output
    shape: M=1024,N=1024,K=1024 PTX stage 2 core
    metric: CUDA-event median latency (us)
    value: 232.448
    source_id: blog-volta-gemm-measurements
    source_locator: primary matched timing table, large_dense PTX stage 2 core, 11 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 input / FP32 accumulation and output
    shape: M=16,N=4096,K=4096 cuBLAS core
    metric: CUDA-event median latency (us)
    value: 83.968
    source_id: blog-volta-gemm-measurements
    source_locator: primary matched timing table, small_m cuBLAS core, 11 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 input / FP32 accumulation and output
    shape: M=16,N=4096,K=4096 PTX stage 2 core
    metric: CUDA-event median latency (us)
    value: 244.736
    source_id: blog-volta-gemm-measurements
    source_locator: primary matched timing table, small_m PTX stage 2 core, 11 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 input / FP32 accumulation and output
    shape: M=127,N=513,K=769 cuBLAS core
    metric: CUDA-event median latency (us)
    value: 50.176
    source_id: blog-volta-gemm-measurements
    source_locator: primary matched timing table, ragged cuBLAS core, 11 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 input / FP32 accumulation and output
    shape: M=127,N=513,K=769 PTX stage 2 core
    metric: CUDA-event median latency (us)
    value: 47.104
    source_id: blog-volta-gemm-measurements
    source_locator: primary matched timing table, ragged PTX stage 2 core, 11 samples
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 input / FP32 accumulation and output
    shape: M=127,N=513,K=769 CUTLASS 2.11.0 Sm70 core
    metric: CUDA-event median latency (us)
    value: 34.816
    source_id: blog-volta-gemm-measurements
    source_locator: interleaved supplemental matrix, ragged CUTLASS core, 11 samples
artifact_dir: artifacts/kernels/volta-gemm/variants
---

# Contract

The checked operator is row-major `A[M,K] * B[K,N]` with represented FP16
inputs, FP32 accumulation and output, `alpha=1`, and `beta=0`. Optional bias
is FP32 and indexed by output column; ReLU follows it. Deterministic cases
cover dense, small-M, and non-tile-multiple dimensions. The CPU oracle reads
the serialized binary16 bits and creates a binary64 reference before any GPU
run. Every output is checked against the predeclared gamma-scaled tolerance.
Because PTX MMA accumulation order and rounding are unspecified, that
tolerance is a fixed admission policy, not a general error theorem.

# Implementations and measured choices

`artifacts/kernels/volta-gemm/variants/` contains the exact native sources.
The library controls are cuBLAS and a genuine CUTLASS `Sm70` `device::Gemm`
instantiation pinned to v2.11.0 commit
`66d9cddc832c1cdc2b30a8755274f7f74640cfe6`, BSD-3-Clause. The custom path
uses explicit `mma.sync.aligned.m8n8k4` and shape-aware tiles. It was compiled
with CUDA 12.9.86 and measured on one V100; CUTLASS documentation describes
CUDA 11.x, so that local compile/run is scoped to this checked configuration.

The custom mainloop packs two adjacent half values per operand and issues the
FP32-accumulating Volta instruction; this is the contiguous call in
`variants/volta_ptx_gemm.cu`:

```cpp
asm volatile(
    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
    "{%0,%1,%2,%3,%4,%5,%6,%7};"
    : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]),
      "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7])
    : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
```

| Case | Shape `(M,N,K)` | PTX tile `(M,N,K)` | cuBLAS p50 ± MAD (us) | PTX stage 2 p50 ± MAD (us) | Shape-specific result |
|---|---:|---:|---:|---:|---|
| Dense | `(1024,1024,1024)` | `(64,64,32)` | `61.440 ± 1.024` | `232.448 ± 1.024` | Use cuBLAS for this tested dense shape. |
| Small-M | `(16,4096,4096)` | `(16,32,32)` | `83.968 ± 1.024` | `244.736 ± 0.000` | Use cuBLAS for this tested small-M shape. |
| Ragged | `(127,513,769)` | `(32,64,32)` | `50.176 ± 2.048` | `47.104 ± 0.000` | PTX/cuBLAS is inconclusive; CUTLASS measured `34.816 ± 0.000` in the interleaved matrix. |

Times are 11-sample unprofiled CUDA-event medians. On ragged, controller
primary timing includes high outliers for both routes. The supplemental
same-GPU rotated matrix has cuBLAS `49.152 ± 2.048 µs` and PTX stage 2
`48.128 ± 1.024 µs`; its paired difference median is `-2.048 µs` with MAD
`2.048 µs`. That small, quantized gap does not justify selecting PTX over the
library. CUTLASS is the measured alternative for this exact ragged case only.

The full matrix and sample vectors are in
[`volta-gemm-measurements`](../../sources/blogs/volta-gemm-measurements.md).
The resulting guidance is bounded: use cuBLAS as the general dense and
small-M route, and consider this pinned CUTLASS instantiation for the tested
ragged contract. It is not a universal dispatch rule.

# Correctness and limits

The independent reference, represented inputs, and contract are stored in
`benchmark_runs/volta-gemm/cases/`. Primary cuBLAS and PTX stage-2 runs checked
all outputs for all three cases with zero failed/nonfinite values and zero
maximum absolute error. Supplemental checks also passed for PTX stage 1,
CUTLASS, cuBLAS plus separate bias/ReLU, and fused PTX bias/ReLU on every
shape. The short exact one-hot compatibility check uses the E09 map; it does
not exercise a multi-tile K-loop, while the full-K case checks exercise the
stage-2 `has_next` path.

Raw memcheck reports zero errors and the native comparison reports pass for
all three shapes, but the controller wrapper retained
`partial`/`conclusive=false` because it expected a different sanitizer stdout
schema. This page does not elevate that wrapper state to a completed
sanitizer gate. Selected Nsight Compute captures are one-kernel samples with
valid counters but invalid replay timing; all three summaries are partial,
classify the kernel as mixed, and request more data. They do not establish a
performance cause. See the source report for exact evidence paths, binary and
library hashes, and measurement limitations.
