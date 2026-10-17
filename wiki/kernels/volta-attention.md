---
id: kernel-volta-attention
title: Volta Attention Prefill, Decode, and Selected-Block Kernels
type: kernel
architectures: [sm70]
tags: [attention, sparse-attention, decode, prefill, software-exp]
confidence: experimental
reproducibility: benchmarked
kernel_types: [attention]
languages: [cuda-cpp]
related: [technique-volta-attention-dataflow, kernel-volta-row-kernels]
sources: [blog-volta-attention-measurements]
performance_claims:
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 represented Q/K/V; FP32 accumulation, probabilities, and output
    shape: prefill Q=(1,4,32,64), KV=(1,2,512,64), causal GQA
    metric: CUDA-event median latency (us)
    value: 97.280
    source_id: blog-volta-attention-measurements
    source_locator: "v1 primary timing role baseline_prefill_timing; prefill64-long-causal; 11 samples; native runner.stdout.txt"
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 represented Q/K/V; FP32 accumulation, probabilities, and output
    shape: prefill Q=(1,4,32,64), KV=(1,2,512,64), causal GQA
    metric: CUDA-event median latency (us)
    value: 96.256
    source_id: blog-volta-attention-measurements
    source_locator: "v1 primary timing role candidate_prefill_timing; prefill64-long-causal; 11 samples; native runner.stdout.txt"
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 represented Q/K/V; FP32 accumulation, probabilities, and output
    shape: decode Q=(1,8,1,70), KV=(1,2,4096,70), causal GQA; materialized cuBLAS path
    metric: CUDA-event median latency (us)
    value: 148.480
    source_id: blog-volta-attention-measurements
    source_locator: "v1 primary timing role baseline_decode_timing; decode70-causal; 11 samples; native runner.stdout.txt"
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 represented Q/K/V; FP32 accumulation, probabilities, and output
    shape: decode Q=(1,8,1,70), KV=(1,2,4096,70), causal GQA; online SIMT path
    metric: CUDA-event median latency (us)
    value: 704.512
    source_id: blog-volta-attention-measurements
    source_locator: "v1 primary timing role candidate_decode_timing; decode70-causal; 11 samples; native runner.stdout.txt"
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 represented Q/K/V; FP32 accumulation, probabilities, and output
    shape: decode Q=(1,8,1,70), KV=(1,2,4096,70), causal GQA; 16-way split partial plus merge
    metric: CUDA-event median latency (us)
    value: 66.560
    source_id: blog-volta-attention-measurements
    source_locator: "v2 primary timing role split16_decode_timing; decode70-causal; 11 samples; runner.stdout.txt; full partial and merge interval"
artifact_dir: artifacts/kernels/volta-attention/variants
---

# Operator and numerical contract

For query row `i` and query head `h`, the implementation computes scores
`(q[h,i] dot k[g(h),j]) / sqrt(D)`, applies the declared masks, performs a
stable softmax, and returns the probability-weighted sum of values. The scale
is applied before softmax. `g(h) = h // (Hq/Hkv)` maps grouped-query heads to
KV heads; each query head has its own maximum, denominator, and output.

Q, K, and V are first represented as IEEE FP16. Dot products, scaled scores,
probabilities, online softmax state `(m,l,u)`, and outputs use FP32. The
independent CPU oracle computes in FP64 from those exact represented FP16
values. The fixed output check is
`abs(actual-reference) <= 1.5e-4 + 2e-4*abs(reference)`; row maxima and
normalizers have separate fixed thresholds in the contract. These are bounded
engineering admission tolerances, not formal error bounds. The default input
domain is `|Q,K,V| <= 0.25`, `D <= 128`, and `Sk <= 4096`. The separate
high-contrast score stress fixture uses `|Q,K| <= 4` and remains outside that
default domain while using the same tolerance.

The seven frozen cases cover D=64 and 128 prefill, ragged D=70 decode,
right-aligned causal masking, GQA, a caller-supplied selected-block list plus
per-query boolean mask, and a separate score-contrast fixture. The native
correctness roles checked all 39,472 outputs across 552 rows. The immutable
represented input, FP64 reference, case index, per-case oracles, and exact
contract are in `benchmark_runs/volta-attention/cases/attention/`.

# Mask and selection semantics

For unequal query and key lengths, causal query row `i` is aligned to key
position `Sk-Sq+i` and can attend through that key position. The contract
requires `Sq <= Sk`. A byte-valued boolean mask uses `1` to exclude an element.
Multiple mask types combine by exclusion. A row with no active elements
returns exactly zero output and a zero normalizer.

The selected-block case consumes only the block indices supplied by the
caller. Indices must be unique and in range; duplicate indices are rejected,
index order does not change membership, the final partial block is clipped to
`Sk`, and an empty selection produces the defined zero row. Causal and
per-query masks still apply inside selected blocks. There is no block planner,
learned selection policy, or native 2:4 sparsity path in this artifact. The
single selected-block correctness case is not evidence for sparse speedup.

# Online update in the measured source

The core update rescales the prior state whenever a new maximum appears. This
excerpt is from `volta_attention_online.cuh`; the surrounding loop applies the
causal, selection, and boolean predicates before consuming a key:

```cuda-cpp
const float score = dot * scale;
const float next_m = fmaxf(running_m, score);
const float old_scale = running_l == 0.0f ? 0.0f : expf(running_m - next_m);
const float new_weight = expf(score - next_m);
running_l = old_scale * running_l + new_weight;
running_u[chunk] = old_scale * running_u[chunk] + new_weight * value;
running_m = next_m;
```

# Implementations and bounded measurements

The materialized baseline uses cuBLAS for QK, keeps the score/softmax and
probability arrays in FP32, expands represented V to FP32, and uses SGEMM for
`P*V`. The timed operator includes V expansion, QK, FP32 softmax, and PV.
Input loading, allocation, and H2D transfer are outside the CUDA-event
interval. The expansion is part of the measured baseline; it is not treated as
a free reusable cache.

The online route is a fused SIMT kernel with FP32 running `(m,l,u)` state. It
is not a Tensor Core attention kernel and makes no fragment-exchange or fused
Tensor Core claim. On the measured long-prefill shape its result is
inconclusive. On one-query D=70 decode it loses substantially to the
materialized library baseline. A 16-way split-KV route computes FP32 partial
state and merges it with a separately launched kernel; both launches and the
entire merge are inside its reported timing. The source has a unique warp-0
merge owner. This route was added after profile evidence showed the unsplit
decode launch used only eight CTAs on an 80-SM V100. It is selected only for
the exact tested decode case; these measurements do not establish a general
dispatch boundary. The runner's case context reserves materialized score,
probability, and expanded-V buffers for all implementations, including online
and split. Those buffers are unused by the online operators, but this harness
does not measure peak-memory or allocation savings. The online operator's
required state is `(m,l,u)` per active query row, plus tile temporaries; split
also stores one such partial per query-row/split before merge.

| Case and route | Median ± MAD (us) | Samples | Result within tested scope |
|---|---:|---:|---|
| `prefill64-long-causal`, materialized baseline | `97.280 ± 1.024` | 11 | Reference route |
| `prefill64-long-causal`, online SIMT | `96.256 ± 0.000` | 11 | No decisive improvement |
| `decode70-causal`, materialized baseline | `148.480 ± 3.072` | 11 | Faster than unsplit online |
| `decode70-causal`, online SIMT | `704.512 ± 1.024` | 11 | Rejected for this measured case |
| `decode70-causal`, 16-way split plus merge | `66.560 ± 0.000` | 11 | About 2.23x faster than baseline for this case |

All measurements are on one Tesla V100-SXM2-16GB, compute capability 7.0,
UUID `GPU-21131915-1488-23af-38dd-1743ae1f5cc8`, with the same immutable
fixtures and native CUDA-event timing procedure. Full sample arrays and role
receipts are linked from
[`volta-attention-measurements`](../../sources/blogs/volta-attention-measurements.md).

# Validation and evidence limits

The v1 baseline and online correctness roles passed all seven cases. The
v1 online memcheck raw log reports zero errors and exit status zero; the
captured native runner reports `status: passed`. Project Control's generic
sanitizer classifier nevertheless retained `partial` and
`conclusive: false` because it observed zero errors and labels that case
unknown, with a conditional cuda-gdb suggestion only if a crash reproduces.
The v2 split16 correctness and native output checks passed the decode case;
its memcheck raw log also reports zero errors and exit zero while the same
generic wrapper classification remains partial. These facts are reported
separately rather than upgrading the wrapper status.

The selected Nsight Compute v1 capture contains one online decode-kernel
sample. Its summary is partial, counters are valid, and replay timing is
invalid. It recorded an eight-CTA grid on an 80-SM GPU, 48 registers per
thread, 0.257% DRAM throughput, and 1.014% SM throughput. This supports a
low-grid occupancy hypothesis when read alongside the unprofiled timing, but
does not measure throughput or identify the full bottleneck. The split route
result includes its merge and has separate unprofiled timing evidence.

The original source and v1 evidence are bound to
`benchmark_runs/volta-attention/evidence/v1/source-closure/source-closure.json`
and the v1 baseline/online ELF SHA-256
`082734ce3e1dbff5180f3751256f60e1cbc9a7e87f53baf563d4ff0cc75090df`. The
corrected split merge source and v2 receipts are bound to
`benchmark_runs/volta-attention/evidence/v2/source-closure/source-closure.json`
and the v2 ELF SHA-256
`a09a8742a13ca8f2d6b2d970ec4a5a3663d1c93bfb485633b70a57ac1be3477b`.
The v2 source adds a single-warp merge ownership guard; v1 measurements
remain bound to their unchanged original source snapshot. Aggregate evidence
records live in `benchmark_runs/volta-attention/summary.json`.
