---
id: blog-volta-gemm-measurements
title: 'Volta FP16-input GEMM: native correctness and measured alternatives'
author: KernelWiki Volta GEMM investigation
url: https://github.com/tumlinso/KernelWiki
source_category: benchmark-blog
architectures: [sm70]
tags: [gemm, mma-sm70]
retrieved_at: '2026-10-17'
---

# Scope and contract

This is a first-party native investigation on one Tesla V100-SXM2-16GB
(`GPU-21131915-1488-23af-38dd-1743ae1f5cc8`). It compares row-major
`A[M,K] * B[K,N]`, represented FP16 inputs, FP32 accumulation and output,
`alpha=1`, `beta=0`, and (where selected) FP32 bias followed by ReLU. Every
route consumed the same serialized IEEE binary16 inputs for its case. The
expected output is an independent CPU FP64 product over those represented
values, generated before device execution.

The per-output admission threshold is the predeclared gamma-scaled FP32
rounding policy in each immutable case contract. It is checked on every
output. PTX does not specify MMA accumulation ordering and rounding closely
enough to make this a universal hardware error bound, so the threshold is an
admission criterion rather than a proof about all inputs.

| Case | Shape `(M,N,K)` | Input SHA-256 | FP64 reference SHA-256 | PTX tile | Useful/executed PTX products |
|---|---:|---|---|---|---:|
| `large_dense` | `(1024,1024,1024)` | `4a10e1084e9a934f1082cf3737f65902734fc9997a51dc8ce3365faa4f354ea7` | `d857fafab808da8ced9f3b64166c2f0377d1ef40e437628d75d4f1cf4b92146a` | `64x64x32` | 100% |
| `small_m` | `(16,4096,4096)` | `ed1bdb4a268e252ce4cc9d1fdfbcf5f636f658cb525766b1f54ec038c8e43ca0` | `2744fe8ced972afb676b16da618fd8da94eef92a66f74b41665f8a7821633209` | `16x32x32` | 100% |
| `ragged` | `(127,513,769)` | `59be4d97102a1ebbdc6653ad41f01cd34dee75636ddcf597af05f592d9aec439` | `e34819dd1772853a600b3b46b601c60b6fefba428be78987ee88e54f066c73a4` | `32x64x32` | 84.94% |

The ragged PTX grid executes tiles padded to `128x576x800`; this is why its
useful-product fraction is lower. The case contracts, immutable inputs,
reference vectors, and CPU oracle records are under
`benchmark_runs/volta-gemm/cases/`.

# Implementations and timing method

The native bundle in `artifacts/kernels/volta-gemm/variants/` contains a
cuBLAS baseline, CUTLASS 2.11.0 Sm70, and an explicit
`mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` implementation. PTX stage 1
and stage 2 keep the same math and shape-specific tiles; stage 2 prefetches
the next tile through ordinary global loads into registers and then stores
those values into the alternate shared-memory buffer. It does not use
asynchronous copies or mbarriers. The exact CUTLASS tag resolves to commit
`66d9cddc832c1cdc2b30a8755274f7f74640cfe6` under BSD-3-Clause. Its source
documentation describes CUDA 11.x; the observed CUDA 12.9 compile and V100
run qualify only this checked instantiation and environment.

Primary matched comparisons used 11 unprofiled CUDA-event samples per route
with the controller's same-SM70-GPU pairing. Values below are medians and
median absolute deviations (MAD), in microseconds. The event interval covers
the selected device operator; allocation and host transfers are outside the
interval. The separate cuBLAS bias/ReLU route includes its GEMM and epilogue
kernels; the fused PTX route includes the full fused kernel. Timing paths are
recorded in `benchmark_runs/volta-gemm/evidence/<case>/`.

| Case | cuBLAS core | PTX stage 2 core | Samples |
|---|---:|---:|---:|
| `large_dense` | `61.440 ± 1.024` | `232.448 ± 1.024` | 11 each |
| `small_m` | `83.968 ± 1.024` | `244.736 ± 0.000` | 11 each |
| `ragged` | `50.176 ± 2.048` | `47.104 ± 0.000` | 11 each |

A second, controller-admitted matrix interleaved 11 one-sample launches for
each implementation and rotated their order by repetition. This matrix adds
stage 1, CUTLASS, and complete bias/ReLU alternatives:

| Case | cuBLAS core | CUTLASS core | PTX stage 1 | PTX stage 2 | cuBLAS + separate bias/ReLU | fused PTX bias/ReLU |
|---|---:|---:|---:|---:|---:|---:|
| `large_dense` | `61.440 ± 1.024` | `120.832 ± 1.024` | `250.880 ± 1.024` | `231.424 ± 1.024` | `77.824 ± 2.048` | `232.448 ± 1.024` |
| `small_m` | `84.992 ± 2.048` | `125.952 ± 0.000` | `561.152 ± 0.000` | `244.736 ± 1.024` | `91.136 ± 1.024` | `245.760 ± 1.024` |
| `ragged` | `49.152 ± 2.048` | `34.816 ± 0.000` | `61.440 ± 0.000` | `48.128 ± 1.024` | `53.248 ± 1.024` | `47.104 ± 0.000` |

The two timing views agree that PTX stage 2 is a large and small-M loss to the
library baselines. On ragged, the PTX/cuBLAS primary medians are close, and
the rotated matrix's paired PTX-stage2-minus-cuBLAS differences have median
`-2.048 µs` and MAD `2.048 µs` (`-9.216` to `+14.336 µs`). The quantized
samples overlap materially, so that comparison remains inconclusive. CUTLASS
is faster on this exact ragged contract (`34.816 µs`) than both cuBLAS and
PTX stage 2, and is retained as the measured shape-specific alternative.
This does not imply a general CUTLASS speedup beyond the tested shape, data
type, layout, tile, toolchain and V100.

Across all three shapes, two-stage PTX improves on the same PTX tile at one
stage; it does not catch either library on dense or small-M. This is a measured
stage-count result, not proof that a different tile or a hardware asynchronous
path would behave the same.

# Epilogue and profile limits

The bias/ReLU PTX consumer uses each accumulator's owning lane and slot, so
producer and consumer ownership are identical (`P(x)=Q(x)`). It removes the
intermediate output and separate launch, but there is no shuffle to remove and
no second MMA in this experiment. The fused PTX path and the materialized
cuBLAS path also use different GEMM engines; their full-operator difference
cannot isolate a fusion-only speedup. The timing table reports those complete
costs without assigning the difference to layout movement.

All three selected Nsight Compute runs sampled the PTX stage-2 GEMM kernel
once. Counter collection was valid, timing was invalid under replay, and the
summary labeled each result partial/mixed and requested more data. These
profiles do not establish a cause or replace unprofiled timing. The native
memcheck logs each show `ERROR SUMMARY: 0 errors`, exit status zero, and a
passing per-output native comparison; the controller wrapper classified the
three records as partial/conclusive-false because the native check uses
`KERNELWIKI-GEMM-CHECK/1` rather than its environment-only PASS schema. The
raw records remain available and are not restated as a completed sanitizer
gate here.

Controller-backed primary and supplemental records, raw event samples,
selected NCU reports, source identities, and exact runner binary are preserved
under `benchmark_runs/volta-gemm/`. The measured runner SHA-256 is
`fd6989c76a9d3925236771e4c7e8a9f70cda5fa2820420d1d1623e8408330ac7`; the
runtime cuBLAS library is `libcublas.so.12.9.1.4`, SHA-256
`d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048`, and
the program queried API version `120901`.

## Ordered frontend continuation

The preserved runner uploads pageable inputs on the default stream and uses a
nonblocking compute stream, leaving a possible ordering race. A separate
[ordered frontend](../../artifacts/kernels/volta-gemm-ordered/variants/README.md)
submits both uploads and computation to the same explicit stream.
Its [fresh correctness records](../../benchmark_runs/volta-gemm-ordered/RESULTS.md)
pass all six routes on all three shapes and both one-hot stage counts. That
new ELF has no timing or profiler result; the historical measurements and raw
records above are unchanged.
