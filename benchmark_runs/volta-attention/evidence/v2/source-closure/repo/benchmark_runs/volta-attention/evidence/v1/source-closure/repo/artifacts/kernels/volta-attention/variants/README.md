# Volta attention source variants

This derived source bundle implements forward scaled dot-product attention for
CUDA compute capability 7.0. It is an operator study, not a serving runtime:
there is no learned selector, cache manager, backward pass, dropout, native
2:4 sparse path, or model integration.

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

`volta_attention_runner.cu` contains a materialized cuBLAS baseline and native
host runner. The baseline computes FP16-input QK with FP32 accumulation and
scores, performs stable FP32 softmax, expands the represented FP16 V values to
FP32, then uses SGEMM for FP32 probabilities times FP32-expanded V. Its full
conversion, materialization, and compute sequence is included in operator
timing.

`volta_attention_online.cuh` contains a fused SIMT online `(m,l,u)` route and a
split-KV partial/merge route. It handles head dimensions through 128, including
ragged D=70 with bounds checks. The split candidate includes both partial
production and merge; it is conditional on measured low-query parallelism.
Neither route claims a Tensor Core fusion or native sparse acceleration.

All copies, kernels, cuBLAS work and timed events use one explicit nonblocking
CUDA stream. Pageable-host setup uploads use `cudaMemcpyAsync` and are drained
before their input vectors can leave scope. Correctness readback is synchronized
before host inspection. Setup, H2D and correctness readback remain outside the
steady-state event window.

The portable FP64 semantic oracle, deterministic represented inputs, tests,
contracts, controller specs and captured evidence live under
`benchmark_runs/volta-attention/`. The binary fixture binds the exact FP16 Q/K/V
representations and FP64 outputs/row statistics used for native comparisons.
