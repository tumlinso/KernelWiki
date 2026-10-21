# Volta attention source variants

This derived source bundle implements forward scaled dot-product attention for
CUDA compute capability 7.0. It is an operator study, not a serving runtime:
there is no learned selector, cache manager, backward pass, dropout, native
2:4 sparse path, or model integration.

## Qualification status

The original `materialized_softmax` implementation in
`volta_attention_runner.cu` is disqualified as a correctness or performance
baseline. After reading the shared row maximum, threads can overwrite the same
shared reduction array with local sums before every reader has finished. A
focused KDA audit of the corresponding materialized control recorded 24
racecheck hazards (4 errors, 20 warnings). Keep this source, its digest, and
the associated v1 evidence unchanged as historical artifacts, but do not use
its timings or comparisons derived from them as qualified claims.

KDA's separately named `materialized-corrected` downstream snapshot adds a CTA
barrier and passed memcheck, racecheck, and synccheck on 24 cases. It is an
independent correction and qualification; it does not validate this source or
the archived v1 baseline measurements. The saved audit is recorded in KDA at
`runs/attention-decode-study/evidence/independent-review/corrected-baseline-control.md`.

## Fixed contract

- Contiguous row-major Q/K/V use `[B,H,S,D]`; input values are represented as
  FP16 before entering either native route.
- Each score is `(q dot k) / sqrt(D)`, with scale applied before stable softmax.
- Query/key dot products, softmax state, probabilities and outputs use FP32.
  The candidate retains `(m,l,u)` in FP32 and does not round probabilities.
- GQA maps query head `h` to KV head `h / (Hq/Hkv)` but each query row keeps
  its own normalization and output state.
- Unequal-length causal inputs are suffix aligned: query row `i` is positioned
  at `Sk-Sq+i`; causal inputs require `Sq <= Sk`.
- Boolean mask bytes use `1` for excluded keys. Optional selected block IDs are
  per query row, unique, in-range, order independent, and combined with causal
  and boolean masks. A final partial block is clipped to `Sk`.
- Empty/all-masked rows produce zero output, maximum `-inf`, and normalizer 0.

## Implementations

`volta_attention_runner.cu` contains the historical materialized cuBLAS route
and native host runner. That route computes FP16-input QK with FP32 accumulation and
scores, performs stable FP32 softmax, expands the represented FP16 V values to
FP32, then uses SGEMM for FP32 probabilities times FP32-expanded V. Its full
conversion, materialization, and compute sequence is included in operator
timing.

`volta_attention_online.cuh` contains a fused SIMT online `(m,l,u)` route and a
split-KV partial/merge route. It handles head dimensions through 128, including
ragged D=70 with bounds checks. The split candidate includes both partial
production and merge; its merge CTA assigns all output/statistic stores to
warp 0, whose 32 lanes cover the supported D<=128 in four feature chunks. The
first frozen v1 build did not execute this route; after its source was frozen,
the duplicate-writer issue was corrected in the v2 source. The split candidate
is only promoted if the versioned native correctness, sanitizer, and full-cost
timing checks pass. Neither route claims a Tensor Core fusion or native sparse
acceleration.

All copies, kernels, cuBLAS work and timed events use one explicit nonblocking
CUDA stream. Pageable-host setup uploads use `cudaMemcpyAsync` and are drained
before their input vectors can leave scope. Correctness readback is synchronized
before host inspection. Setup, H2D and correctness readback remain outside the
steady-state event window.

The portable FP64 semantic oracle, deterministic represented inputs, tests,
contracts, controller specs and captured evidence live under
`benchmark_runs/volta-attention/`. The binary fixture binds the exact FP16 Q/K/V
representations and FP64 outputs/row statistics used for native comparisons.
