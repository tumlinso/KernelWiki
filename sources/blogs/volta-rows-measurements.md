---
id: blog-volta-rows-measurements
title: 'Volta FP32 row operators: softmax and RMSNorm measurements'
author: KernelWiki Volta row-kernel investigation
url: https://github.com/tumlinso/KernelWiki
source_category: benchmark-blog
architectures: [sm70]
tags: [reduction, mma-sm70, vectorized-loads]
retrieved_at: '2026-10-17'
---

# Scope and numerical contract

This native investigation compares stable FP32 row softmax and plain FP32
RMSNorm on one Tesla V100-SXM2-16GB (`GPU-21131915-1488-23af-38dd-1743ae1f5cc8`,
compute capability 7.0). The fixed bundle has 22 cases covering both operators,
widths from 31 through 8193, tails, row strides, masks, all-masked rows,
misalignment, zero-valued RMSNorm rows, zero row counts, one wide row, and
streaming rows. Correctness and memcheck target the full bundle.

The suite consumes one immutable represented FP32 input bundle and an
independently computed binary64 reference over those decoded FP32 values:

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| `cases/rows/inputs.bin` | `78ca394b1441b60b59a283be943301c21488d7ee7a864643c445c4ed3793d909` | 17,186,335 |
| `cases/rows/reference.f64le` | `d483d4110b4414d5bbfbf27a0e9976ead21152ca9464d455d7ae6b6ef9b45d43` | 34,228,484 |
| `cases/rows/cpu-oracle.json` | `3ef95dbcfeb88ccb8609838e79580ac27ffe3b61aa137add0ad2c86a3c366857` | 7,117 |
| `cases/rows/contract.json` | `816fd497f250e912c4a1dc1b7256e199a5ceed385db2748aa16f3eda0426944f` | 18,461 |

The frozen per-output admission rule is
`abs(actual - reference) <= 2^-30 + 2^-14 * abs(reference)`. Softmax accepts
finite logical inputs in `[-16,16]`, optional byte masks containing `0` or
`1`, returns exactly zero for masked outputs, and returns an all-zero row when
every element is masked. RMSNorm computes
`gamma * x * rsqrt(epsilon + sum(x*x)/N)` with `abs(x) <= 64`,
`abs(gamma) <= 4`, and `epsilon` in `[1e-5,1e-2]`. Width is `1 <= N <= 8193`;
zero rows are valid and skip launch. Independent input/output/mask strides
ignore input padding and preserve output padding. RMSNorm has no mask,
residual, bias, or fused producer in this contract.

Both baseline and candidate native comparisons passed all 22 cases and checked
4,278,449 outputs, with zero failed, non-finite, or padding-write outputs.
Per-case errors, declared routes, and checker identities are retained in
`benchmark_runs/volta-rows/summary.json` and the raw native JSON under
`benchmark_runs/volta-rows/evidence/v2/roles/`.

# Implementations and timing method

The baseline is a warp-per-row softmax mapping (four rows per 128-thread CTA)
and scalar one-CTA-per-row RMSNorm. The candidate is one CTA per softmax row,
and a one-CTA RMSNorm path with guarded `float4` loads plus scalar fallback.
All participating lanes execute synchronized warp shuffles; tail and mask
elements contribute the reduction identity. CTA reductions use shared partials
and barriers. The vector RMSNorm route checks actual post-offset input, gamma,
and output addresses and row-stride alignment before issuing vector loads; it
handles the tail scalarly and falls back when alignment is unsupported.

Controller-admitted timing runs use five warmups, 21 CUDA-event samples per
case, and two alternating input/output buffer sets. Both implementations use
the same three case IDs. Setup copies and allocation are outside the event
interval. Medians and median absolute deviations (MAD) are in microseconds:

| Case | Shape / path | Baseline p50 ± MAD | Candidate p50 ± MAD | Candidate/baseline |
|---|---|---:|---:|---:|
| `softmax_wide_single_row` | 1 × 4096; warp → CTA | `24.576 ± 0.000` | `11.264 ± 0.000` | `0.458` (`−54.2%`) |
| `softmax_wide_streaming` | 512 × 4096; warp → CTA | `92.160 ± 1.024` | `47.104 ± 1.024` | `0.511` (`−48.9%`) |
| `rmsnorm_vec4_wide_streaming` | 512 × 4096; scalar → aligned `float4` | `44.032 ± 1.024` | `36.864 ± 1.024` | `0.837` (`−16.3%`) |

Each result is bounded to this V100, fixed input, source, and tested shape.
The two softmax comparisons favor the CTA route for these workloads; the
RMSNorm comparison favors the aligned vector route for this streaming case.
The misaligned correctness case selects the scalar fallback. These three
points do not define a general dispatch threshold or establish why a route is
faster. Full 21-sample vectors and per-case sizes are in the summary and role
stdout files.

The device reported L2 capacity as 6,291,456 bytes. The rotated total
input/output working set was 65,536 bytes for the single-row softmax,
33,554,432 bytes for streaming softmax, and 33,587,200 bytes for streaming
RMSNorm. The latter two exceed L2 capacity. That size comparison alone does not
prove which requests reached HBM or how many bytes were transferred.

# Profile and validation limits

One candidate `softmax_wide_streaming` launch was selected for Nsight Compute.
The profile found the `softmax_cta(...)` kernel and classified its one sample as
memory-bound, reporting 44.01% DRAM throughput, 20.30% SM throughput, and 32
registers per thread. Occupancy was unavailable. The profile summary is
`partial`, counters are valid, replay timing is invalid, and it requests more
data. It can inform a follow-up measurement, but it does not explain the
unprofiled timing delta and does not characterize RMSNorm.

The matched candidate ELF's memcheck wrapper summary remains
`partial`/`conclusive=false`, with exit status zero and zero reported errors.
The preserved raw log says `ERROR SUMMARY: 0 errors`, and its captured native
full-suite output passes all 22 cases. The native result and raw error count
are evidence; the wrapper classification is not restated as a full sanitizer
pass.

An earlier v1 correctness attempt is retained as a failure in
`evidence/v1-failure/manifest.json`. It failed on two large cases. The frozen
v2 runner uses stream-ordered asynchronous setup copies and drains setup before
event timing; the tolerances and inputs were unchanged. The failed v1 record
remains diagnostic and is not included in the v2 acceptance.

The published source copy is under `artifacts/kernels/volta-rows/variants/`;
the byte-identical pre-execution source snapshot is under
`benchmark_runs/volta-rows/evidence/v2/source-snapshot/full/`. Controller role specifications and evidence are under
`benchmark_runs/volta-rows/`, and the aggregate source/input/binary/lease
bindings are in `benchmark_runs/volta-rows/summary.json`. The candidate ELF
measured by correctness and subsequent candidate roles has SHA-256
`229b3a572c42b2db8237c86ae71bd9bb4d9d8cd06384be29913a8fe7b33a4095`; the
baseline ELF is
`6da764e5a52ab11271c0455cbfd014e815a38ed69ba77c25ba77e45b0ff7bb46`.
