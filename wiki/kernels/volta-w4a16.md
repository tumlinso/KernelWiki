---
id: kernel-volta-w4a16
title: Volta W4A16 Software Decode and Dense Dot Product
type: kernel
architectures: [sm70]
tags: [gemv, gemm, decode, quantization, fine-grained-quantization]
confidence: experimental
reproducibility: benchmarked
kernel_types: [gemv, gemm, decode, quantization]
languages: [cuda-cpp]
related: [kernel-volta-gemm, technique-volta-grouping-and-decoding]
sources: [blog-volta-grouped-lowbit-measurements]
performance_claims:
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 activation and decoded weights; FP32 accumulation/output
    shape: M=1, K=4096, N=1024; direct signed-nibble decode beside SIMT dot
    metric: Median resident CUDA-event batch latency (us)
    value: 809.984
    source_id: blog-volta-grouped-lowbit-measurements
    source_locator: "Native results > W4A16 decode and reuse results > weighted 1,4096,1024 > direct batch median event"
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 decoded weights; dense cuBLAS batch, predecode measured separately
    shape: M=1, K=4096, N=1024; software W4A16 predecode path
    metric: Median resident dense-batch CUDA-event latency (us)
    value: 27.648
    source_id: blog-volta-grouped-lowbit-measurements
    source_locator: "Native results > W4A16 decode and reuse results > weighted 1,4096,1024 > dense batch median event"
artifact_dir: artifacts/kernels/volta-grouped-lowbit/variants
---

# Exact storage contract

This kernel page describes one software W4A16 storage example, not native
four-bit tensor arithmetic. Codes are signed two's-complement integers
`q ∈ [-8,7]`. A row-major logical `[K,N]` tensor packs its even element in the
low nibble and its odd element in the high nibble; unused padding bits after an
odd element count are ignored. One positive finite FP32 scale applies to each
128-element K group and output column.

For element `(k,n)`, decode exactly as
`w16 = round_fp16_rn(round_fp32(scale[floor(k/128),n] * q[k,n]))`. Activations
are represented FP16, products accumulate in FP32, and outputs are FP32. Both
the CPU FP64 oracle and native paths consume the exact same represented
activations, nibble bytes, and scale bytes. The bounded test scale interval is
`[2^-14,8000]`, which keeps all sixteen codes finite in FP16.

The materialized decode path converts each scaled signed code explicitly to
round-to-nearest FP16 before storing the dense matrix:

```cuda
const int q = signed_code(packed_codes, index);
const float scaled = scales[static_cast<std::size_t>(group) * n + col]
                   * static_cast<float>(q);
const __half represented = __float2half_rn(scaled);
decoded_bits[index] = __half_as_ushort(represented);
```

## Measured decode boundary

Across all four V100 cases, materializing FP16 weights and using dense cuBLAS
had lower resident device-event time than the direct SIMT decoder. In the
weighted `K=4096,N=1024` cases, the measured predecode event median was 45.056 µs;
the dense batch medians were 27.648 µs (`M=1`) and 31.744 µs (`M=16`), while
direct decode measured 809.984 and 772.096 µs. The corresponding one-use
prep-plus-compute device totals, calculated by adding separate medians, are
72.704/76.800 µs for materialize+dense versus 809.984/772.096 µs direct.
These sums are not directly measured end-to-end device intervals. Direct
decode saves the dense expansion but has no throughput win in this
implementation and regime.

The derived setup-plus-batch host model also includes input and FP64-oracle
I/O, validation, allocations, H2D, and synchronization. It adds measured
predecode host time and a median repeated-compute batch; it is therefore an
estimate under the harness with an initialized CUDA context, not production
startup latency. Reuse
holds each case's activations, packed weights, and scales unchanged. The
`M=1` and `M=16` weighted fixtures share byte-identical packed weights/scales,
but their activation values are independently generated rather than prefix
matched. The full measured-component and derived-cost tables are in the
[measurement report](../../sources/blogs/volta-grouped-lowbit-measurements.md)
and [`summary.json`](../../benchmark_runs/volta-grouped-lowbit/summary.json).

# Decode placement

The materialized route decodes the packed matrix to FP16 and passes it to
cuBLAS. The direct candidate reconstructs decoded values beside each SIMT dot
product and is intended for very small M. Reuse batches compare actual repeated
compute calls against the same unchanged packed weights, scales, and
activations. Reports separate the measured predecode median, dense or direct
compute batch, and initial host/device preparation. Prep-plus-batch values are
derived sums of those measurements, not directly measured end-to-end intervals.

The weighted `K=4096,N=1024` case contains 2,097,152 bytes of packed codes
plus 131,072 bytes of FP32 scales (2.125 MiB) versus 8,388,608 bytes (8 MiB)
for expanded FP16 weights. This describes representation capacity only; it
does not establish HBM or L2 residency. Results and applicability limits are
in the [measurement report](../../sources/blogs/volta-grouped-lowbit-measurements.md).

# Format boundary

The case does not claim native INT4/NVFP4 compute, compatibility with a model
checkpoint, or a universal quantization scale convention. See
[`volta-grouping-and-decoding`](../techniques/volta-grouping-and-decoding.md)
for the cost and numerical rules that govern decode placement on Volta.
