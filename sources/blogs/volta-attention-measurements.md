---
id: blog-volta-attention-measurements
title: 'Volta attention: prefill, decode, and selected-block measurements'
author: KernelWiki Volta attention investigation
url: https://github.com/tumlinso/KernelWiki/blob/master/benchmark_runs/volta-attention/summary.json
source_category: benchmark-blog
architectures: [sm70]
tags: [attention, sparse-attention, conditional-rescaling, software-exp]
retrieved_at: '2026-10-17'
---

# Post-study qualification update (2026-10-21)

The original v1 `materialized_softmax` source is disqualified as a correctness
or performance baseline. It reads the CTA-shared maximum and then reuses the
same reduction array for per-thread sums without a barrier between those
operations. A focused KDA racecheck audit of the corresponding materialized
control reported 24 hazards (4 errors, 20 warnings). The original materialized
timing rows below remain as historical records, but they are not qualified
benchmark claims. Withdraw the prefill comparison, the decode comparison, and
every speedup or slowdown that uses those values.

KDA separately evaluated a `materialized-corrected` snapshot with a CTA barrier
at that boundary. That downstream snapshot passed memcheck, racecheck, and
synccheck over 24 cases; it is a distinct candidate and does not retroactively
validate the original KernelWiki source, its run, or any v1 baseline-relative
claim. The saved KDA audit is at
`runs/attention-decode-study/evidence/independent-review/corrected-baseline-control.md`;
its focused sanitizer summary is at
`runs/attention-decode-study/evidence/independent-review/baseline-sanitizer-audit.txt`.

The source closures, binaries, role receipts, and timing samples in this
submodule remain frozen archive evidence. The evidence checker verifies their
captured identities and integrity; it does not qualify the materialized
implementation after this safety finding.

# Scope, represented data, and contract

This first-party native investigation compares a materialized cuBLAS-backed
attention route, an online SIMT route, and one conditional 16-way split-KV
route on a Tesla V100-SXM2-16GB (compute capability 7.0,
`GPU-21131915-1488-23af-38dd-1743ae1f5cc8`). It covers prefill, one-query-row
decode, and supplied selected blocks. It is not a general attention library or
a broad dispatch study.

All routes consume one immutable bundle of represented FP16 Q/K/V. The
independent CPU oracle uses FP64 arithmetic over the exact decoded half
values. The binary bundle and fixed contract are:

| Artifact | SHA-256 |
|---|---|
| `cases/attention/attention_inputs.bin` | `1fed25c14ebae3c5d41bc5820dafcd881637a95344772cdfa6c730c9318eef5d` |
| `cases/attention/attention_reference.bin` | `d1f9cf4ebe78311f336970b7e3f91df9aa921ed191c6009a653ea663b931b969` |
| `cases/attention/attention_contract.json` | `d66dc00a131035b99f6386250bf75ebaaaea26f9cbf66d27eb3c44ce22ceb810` |
| `cases/attention/case-index.json` | `9b90922ebc789a010ba8017084870be5c4da5cdcfb56645c4f69d0398b89af8d3` |

The contract applies `1/sqrt(D)` before softmax. Native QK accumulation,
softmax probabilities, online `(m,l,u)` state, and output are FP32; Q/K/V are
FP16. Query heads map to KV heads by `h//(Hq/Hkv)` for GQA while retaining
independent row normalizers. For non-square causal input, query row `i` is
right-aligned to key position `Sk-Sq+i`, with `Sq<=Sk`. Byte mask value 1
excludes. Selected block ids are caller-supplied, unique, and in-range; order
does not change membership, the tail is clipped, duplicate ids are rejected,
and an empty selection yields zero. A fully excluded row also yields zero
output and zero normalizer.

The predeclared per-output tolerance is
`abs(actual-reference) <= 1.5e-4 + 2e-4*abs(reference)`, with separate fixed
row-maximum and normalizer checks. These are engineering admission envelopes,
not formal bounds. The bounded default input domain is `|Q,K,V|<=0.25`,
`D<=128`, and `Sk<=4096`. The `score-stress64-noncausal` fixture uses
`|Q,K|<=4`, is explicitly outside that default domain, and retains the same
fixed tolerance. The seven cases are `prefill64-causal`,
`prefill64-medium-causal`, `prefill64-long-causal`, `prefill128-noncausal`,
`decode70-causal`, `selected64-causal`, and `score-stress64-noncausal`.

# Implementations and timing boundary

The materialized baseline uses cuBLAS for QK, stores scores and probabilities
in FP32, performs stable FP32 softmax, expands the represented FP16 V values
to FP32, then uses SGEMM for FP32 `P*V`. V expansion, QK, softmax, and PV are
included in the timed implementation. Allocation, host input loading, and
H2D transfer are excluded. The measured path does not assume an already
expanded V cache.

The online implementation is a fused SIMT kernel with FP32 running state. It
is neither a Tensor Core attention route nor a native 2:4 sparse route. A
separate split route computes 16 partial `(m,l,u)` states and launches a merge
kernel. Its timing includes both partial and merge work. The initial split
merge wrote output redundantly from four warps; the corrected v2 source makes
warp 0 the unique owner. v1 captures remain attached to the original v1
source and binary.

Each timing role collected 11 unprofiled CUDA-event samples for one case.
Values are the sample median and median absolute deviation (MAD), in
microseconds:

| Case | Route | Median ± MAD (us) | Source role | Native case and check |
|---|---|---:|---|---|
| `prefill64-long-causal` | materialized baseline (disqualified; historical only) | `97.280 ± 1.024` | `baseline_prefill_timing` | `prefill64-long-causal`, `checks[0]`, 8,192 outputs / 128 rows |
| `prefill64-long-causal` | online SIMT | `96.256 ± 0.000` | `candidate_prefill_timing` | `prefill64-long-causal`, `checks[0]`, 8,192 outputs / 128 rows |
| `decode70-causal` | materialized baseline (disqualified; historical only) | `148.480 ± 3.072` | `baseline_decode_timing` | `decode70-causal`, `checks[0]`, 560 outputs / 8 rows |
| `decode70-causal` | online SIMT | `704.512 ± 1.024` | `candidate_decode_timing` | `decode70-causal`, `checks[0]`, 560 outputs / 8 rows |
| `decode70-causal` | split16 plus merge | `66.560 ± 0.000` | `split16_decode_timing` | `decode70-causal`, `checks[0]`, 560 outputs / 8 rows |

All five values were recorded in the matching role's `runner.stdout.txt`, and
each timing check carries the full 11-element `samples_us` array. The two
materialized values are disqualified by the later racecheck finding despite
their original runner status and passing numerical checks. Therefore the
prefill difference, the reported 2.23x split16 gain, and the reported 4.75x
online slowdown are withdrawn. The online and split rows remain absolute
historical captures; use only comparisons against independently qualified
baselines. The measured split result is bounded to this one D=70,
one-query-row case and establishes no general dispatch rule.

## Correctness and selected-block scope

The v1 baseline and online full-bundle native output checks passed all seven
cases: 39,472 outputs across 552 rows per implementation. This numerical
check did not detect the baseline's shared-memory race. The CPU reader and
oracle suite exercises binary layout, case decoding, and independent output
checks. The selected-block case checks supplied membership, duplicate
rejection, mask combination, tail clipping, order invariance, and empty-row
semantics; its correctness pass does not show a speedup. There is no block
planner, pruning policy, or 2:4 sparse path.

The v1 online and v2 split16 native correctness outputs passed their respective
checks. Compute Sanitizer memcheck raw logs for both versions record exit zero,
zero errors, and the captured native runner's pass. The generic Project
Control sanitizer classifier retains `partial`/`conclusive=false` because it
classifies “zero errors” as an unknown crash state and suggests cuda-gdb only
if a crash continues to reproduce. This wrapper classification remains
visible; raw memcheck and native app evidence are reported separately.

# Profile interpretation

One Nsight Compute capture selected the v1 `online_rows` decode kernel. NCU
2025.2.1.0 recorded one sample from a grid of eight CTAs on a device with 80
SMs, 48 registers per thread, 0.256876% DRAM throughput, and 1.013819% SM
throughput. Its summary is partial, counters are valid, timing is invalid,
and it requests more data. The small grid gives a plausible occupancy
explanation for the poor unsplit decode result when considered together with
unprofiled event timing. The capture alone neither measures throughput nor
isolates the full cause. The split result is a separate unprofiled run and
includes its merge.

# Immutable provenance and raw evidence

The v1 baseline/online experiment used source manifest SHA-256
`d38b1b2ca178f57d3bdfb1f6062fe1082489a50cbbdde87fa92c54200b9d7626`, source
runner SHA-256
`ef4e590764978d9ebbd4994f4ce43d3c9cff67bf875dadfe5d74a1f84accff83`, header
SHA-256
`2b88a805c26585f52a502e4a2272026926191585a2a823bffb861db78e91d198`, and ELF
SHA-256
`082734ce3e1dbff5180f3751256f60e1cbc9a7e87f53baf563d4ff0cc75090df`. The
complete 51-input source closure is
`benchmark_runs/volta-attention/evidence/v1/source-closure/source-closure.json`
(SHA-256
`8ab2b5f5b82e6ecb1ea8e06daf3e3da238beb6270271fdbc7f0571e3edb77ce9`). Its
role archives are under `benchmark_runs/volta-attention/evidence/v1/roles/`.

The corrected v2 split source manifest SHA-256 is
`b5b0fa8d03d73090175753f85515335e52279da98e90b838a25a134fa2668ac9`; its
source closure contains 107 inputs and is
`benchmark_runs/volta-attention/evidence/v2/source-closure/source-closure.json`
(SHA-256
`1e5b38a333671ed37c752591cef379d2a25e9f2f9f706f7c64033604a92270a7`). The
v2 ELF SHA-256 is
`a09a8742a13ca8f2d6b2d970ec4a5a3663d1c93bfb485633b70a57ac1be3477b`; all
three split16 roles use this same binary. Their archives are under
`benchmark_runs/volta-attention/evidence/v2/roles/`. The v2 change fixes
merge ownership only. Original v1 source, specs, receipts, and archived
closure are preserved without reinterpretation.

The aggregate run manifest is
`benchmark_runs/volta-attention/summary.json`; it binds input and oracle
fixtures, case index, source versions, binaries, GPU lease, native output,
and individual role archives. Baseline runtime identities include
`libcublas.so.12.9.1.4` SHA-256
`d63456d94f1b8e2408a8a9538c8740eb474c9eba48a84f420b15e2e10fbcd048` and
`libcublasLt.so.12.9.1.4` SHA-256
`96ddf4891ca4f4ed1475966783a9185df2abd64a81c3cc713b96d036696fc541`.
