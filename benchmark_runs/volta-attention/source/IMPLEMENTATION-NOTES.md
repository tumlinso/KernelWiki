# KWV-06 scratch design

Scope: forward inference attention only. No backward, dropout, serving scheduler, learned selector, native 2:4 path, or paper-derived speed claims.

## Fixed operator contract

- Row-major tensors `[B,H,S,D]`; Q/K/V values are quantized to binary16 before the oracle sees them.
- `Hq` is divisible by `Hkv`; query head `h` reads KV head `h // (Hq/Hkv)`. Every query head owns an independent softmax state/output.
- Dot products accumulate in FP32 in native paths. The semantic oracle uses FP64 over the exact represented half values. Multiply each dot by `1/sqrt(D)` before softmax. Running `(m,l,u)` and output stay FP32. Do not round probabilities to half in the baseline or candidate.
- Finite inputs, no dropout. Boolean masks use `True = masked`. A row with no valid keys produces zero output and `(m,l)=(-inf,0)`; avoid `-inf - -inf`.
- For causal attention with unequal lengths, query row `i` is aligned to key position `Sk-Sq+i`; require `Sq <= Sk`. Thus `Sq=1` decode is the last position in the supplied KV prefix. Noncausal prefill is separate.
- Selected-block IDs are supplied by the caller per query row. IDs identify `ceil(Sk/block_size)` blocks; final blocks may be short. Duplicates and out-of-range IDs are rejected, block order is immaterial, and an empty list yields zero. Query masks and causal bounds still filter keys inside selected blocks.

## Implementation comparison

1. **Materialized library baseline:** use `cublasGemmEx` with represented FP16 Q/K, FP32 scale/compute, and FP32 score output; apply scale before stable softmax and compute probabilities in FP32. Expand V from the exact represented FP16 values to FP32 and use SGEMM for `P*V`. Include V expansion, setup and the full score/probability intermediates in operator time unless an explicitly documented caller contract reuses expanded V. This preserves FP32 probabilities and avoids rounding probabilities to half. The cuBLAS API documents the FP16-input/FP32-compute/FP32-output GEMM combination; the exact runtime algorithm and SM70 dispatch still require native evidence ([cuBLAS API](https://docs.nvidia.com/cuda/archive/11.7.0/cublas/index.html)). Record runtime/library identities and exact call types.
2. **Fused candidate:** stream K/V tiles and update FP32 `(m,l,u)` with the stable rescale identity. Prefill removes the full score/probability matrices; compare full operator time and workspace against the materialized path. The initial simple candidate is SIMT, so do not claim Tensor Core use. For D=64/128, process the reduction over all D and keep the output state in FP32; ragged D=70 is a correctness/fallback case with explicit bounds.
3. **Low-query decode candidate:** partition the full KV interval into several chunks, compute one `(m,l,u)` partial per query/head/chunk, then merge all partials with the same stable identity. The merge kernel and partial-state traffic are part of latency. Test several chunk counts only if the unprofiled/profile evidence says the unsplit grid has too few independent CTAs. Never omit QK dimensions or KV history.
4. **GQA candidate (conditional):** the initial online kernel maps each query head independently to its shared KV head and preserves independent masks, `m/l`, and outputs. Add grouped-query ownership only after measurements show repeated K/V traffic is material. Count packing and unpacking unless an upstream producer already retains that layout. Do not test both grouped-query and split-KV variants automatically; choose the one that addresses the observed bottleneck.
5. **Sparse selected-block case:** run the same fused row contract with supplied block IDs and compare to a dense boolean-mask reference generated from those exact IDs. This validates membership, causality, empty rows, duplicates, and short final blocks. It contains no selector/planner. Do not claim lower latency absent a measured load-reuse candidate including index handling.

## Evidence and decisions

Correctness cases include small exact FP64 checks for D=64, D=128, and ragged D=70; causal/noncausal; GQA; empty/all-masked rows; mixed signs; near ties and high finite contrasts; changed input buffers; partition-boundary metamorphism; selected-block membership and reordered IDs. Keep the CPU oracle and deterministic FP16 bundle independent of CUDA. Before native work, the task must bind source, exact binary, input bundle, oracle, contract, GPU UUID/SM70, toolchain, controller receipts, sanitizer, unprofiled samples, and the selected report through `scripts/volta_evidence_check.py --case attention`.

The fixed pre-run tolerance policy covers finite `abs(Q,K,V) <= 0.25`, `D <= 128`, and `Sk <= 4096`. At the edge, `sum(abs(q*k)) <= 8` and the scaled score magnitude is at most `8/sqrt(128) ~= 0.7072`; the FP32 `gamma_128` dot-accumulation estimate after scaling is about `5.4e-6`, before the small reciprocal-square-root error. The largest possible score difference is about `1.4144`, so the default-domain exponent arguments stay in `[-1.4144,0]`. FP32 `gamma_1024 ~= 6.1e-5` gives the order of the serial accumulation term in one of four online KV partitions at `Sk=4096`; the stable update also rounds its rescale, exponential and weighted-value operations. The 2-ULP `expf` figure in the CUDA 12.9 guide is based on extensive but non-exhaustive tests and is expressly not a guarantee, so the following are fixed engineering admission thresholds rather than a claimed formal worst-case bound: compare every output with `abs_error <= 1.5e-4 + 2e-4*abs(reference)`, each row maximum with absolute error `<= 2e-5`, and each nonempty row normalizer with `abs_error <= 1e-7 + 5e-4*abs(reference)`; empty rows require exact zero output and zero normalizer. The QK-induced output sensitivity alone is below roughly `3e-6` for the stated V range; the broader output envelope leaves room for FP32 weighted-state accumulation/division and distinct library reduction orders. The deterministic score-stress fixture deliberately has `abs(Q,K)=4`, outside the bounded default domain, to exercise near-tied winners and a large finite score contrast. It uses the same fixed thresholds as a separate stress case; it does not broaden the stated domain. This policy is fixed before native collection and will not be enlarged after a failure. If native errors exceed it, investigate the implementation, math-library behavior and baseline call path before revising the domain or policy. See NVIDIA's [CUDA 12.9 single-precision function accuracy table](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html) for the observed `rsqrtf`/`expf` errors and qualification.

The host runner reads one binary input bundle and its hash-bound FP64 reference bundle. `--mode correctness` checks one implementation; `--mode time` repeats the selected full implementation after correctness and emits one sample list per shape; `--mode single` requires one case, checks it first, and then launches one additional invocation for a profiler selection. The latter does not isolate profiling to a single CUDA kernel by itself: NCU launch selection must account for the preceding correctness invocation and the selected implementation's internal launches. Native output uses `status`, raw-hex `gpu_uuid`, `compute_capability`, case checks, error maxima and failure details, and only writes `status: passed` after the output, row-max and normalizer checks complete.

Do not infer the winning decode candidate in advance. Record whether the measured grid, KV loads, softmax state, split merge or conversion/materialization cost motivates the next test. Report measured and unmeasured shape regimes separately.

## Frozen v1 measurements and conditional split follow-up

The v1 measured executable is archived at
`benchmark_runs/volta-attention/evidence/v1/binaries/volta_attention_runner`
(SHA-256 `082734ce3e1dbff5180f3751256f60e1cbc9a7e87f53baf563d4ff0cc75090df`).
It was built once for SM 7.0 and all baseline, online, sanitizer, timing, and
profile roles reuse those exact bytes. The per-role native stdout, controller
receipt, Project Control export, and lease are archived under
`benchmark_runs/volta-attention/evidence/v1/roles/`.

On the single frozen `prefill64-long-causal` timing case, 11 unprofiled CUDA
event samples gave medians of 97.280 us for the materialized cuBLAS baseline
and 96.256 us for the online SIMT candidate. This is not a decisive prefill
advantage. On `decode70-causal` (Sq=1, Hq=8, Sk=4096, D=70), medians were
148.480 us for baseline and 704.512 us for online SIMT. The measured online
route is therefore retained as a correctness implementation, not the decode
choice.

The selected NCU report is diagnosis-only: its one `online_rows` sample has an
8-CTA grid of 128-thread blocks on an 80-SM V100, with 48 registers per thread.
The reported average DRAM and SM throughput are 0.257% and 1.014% of peak.
NCU itself marks timing invalid and requests more samples; its replay duration
is not an operator timing result. The eight-CTA launch establishes a concrete
underfilled grid, while the very low counters and unprofiled slowdown do not
isolate the only cause. The bounded follow-up is one 16-way KV split (128
partial CTAs) with a separate merge kernel, including both kernels in the
same CUDA-event interval. A split source fix assigns merge outputs to warp 0
only; the frozen v1 source and executable remain archived, and the v2 split
route is not accepted unless its own correctness, sanitizer, and timing roles
pass. No split result is claimed until those roles are captured.
