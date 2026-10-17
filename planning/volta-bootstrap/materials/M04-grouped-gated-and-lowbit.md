# M04 — Grouped math and compressed storage with honest arithmetic

## What transfers from modern KernelWiki examples

Grouped GEMM, gated projections and compressed weights remain useful operations on V100. CLC, TMEM and native FP8/NVFP4 tensor instructions do not. The port therefore changes scheduling, operand preparation and accumulation without changing the operator. Reuse M01 matrix/layout work; do not build a second GEMM engine. [KW-PRIMER, AT-R04, SK-TENSOR]

## Grouped/gated operator

For group e, define `U_e = X_e W_up,e`, `G_e = X_e W_gate,e`, and `Y_e = silu(G_e) * U_e`, with the exact activation and rounding policy recorded. Some callers need only grouped GEMM; retain that simpler case. Input-to-expert assignments, repeated rows and output positions are semantically meaningful. Empty groups, duplicate expert choices and skewed group sizes must be preserved. This is not a distributed MoE router.

### Experiment Q1 — Shared input reuse versus excessive live outputs

Compare separate optimized projections plus an elementwise epilogue against a paired/grouped implementation. Candidate gains are shared input/metadata reuse, fewer launches, and elimination of gate/up intermediates. Candidate costs are doubled accumulator live state, synchronization and loss of the best library tile.

Try one layout that computes both projections together and one that preserves a cheaper boundary. Profile the actual gate/up epilogue, register/spill footprint, input traffic and tail utilization. Use a balanced group set and a deliberately skewed set with a few empty/tiny groups. Do not impose hundreds of distributions.

**Falsifier:** a fused path saves launches but spills enough to lose, or relies on reordering/deduplicating semantically distinct rows. Keep a split fallback for large/awkward groups. Nonlinear approximation changes receive a separate numerical test.

### Experiment Q2 — Is scheduling overhead actually recoverable?

First compare ordinary grouped/bucketed tiles with batched library calls. Only if the observed tail/skew dominates, test a finite work-queue/persistent-grid variant. The complete queue/metadata cost is included. There is no general persistent-kernel win: atlas E16's resident sample was slower than batching/graphs. [AT-E16]

Use independent output tiles with bounded termination and no oversubscribed global spin barrier. This is a local scheduling experiment, not a resident runtime framework. Stop when a simple tile/bucket choice removes the relevant tail or when queue cost remains larger than the saved idle work.

## A precise low-bit starting contract

Implement one **W4A16 storage** case, not every checkpoint format. The proposed teaching/benchmark format is signed two's-complement four-bit weights `q in [-8,7]`, with a separately stored FP32 scale for each documented K-axis group. Define the decoded weight as `w_half = round_fp16(scale * q)`. Activations are FP16, the dot-product accumulates in FP32 and the output conversion is explicit. Choose a group size such as 128 for the initial representative case; a tail group must still be correct.

This is deliberately **not** NVFP4, MXFP4, E4M3 or E5M2. Those formats have different value maps and scale rules. Add an exact existing format only when adapting a relevant pinned source implementation and its independent decoder. Never silently replace its codebook, global scale, zero-point or rounding sequence. [AT-R04, AT-R15, CAND-1CAT-666]

### Experiment Q3 — Decode near the consumer or materialize once?

Compare:

- Decode into a reusable FP16 buffer, then run the best supported GEMM.
- Decode in registers/shared staging immediately before the native multiply.
- For very small M, a direct SIMT/vector GEMV alternative when useful tile occupancy is poor.

The comparison includes weight preparation and each per-use decode cost. A predecoded cache is a different memory-capacity/reuse contract and must be reported separately. Vary small M versus moderate M and reuse of a fixed weight matrix; keep the same represented weights. Inspect whether the limiting resource moved from memory/request traffic to conversion, issue or registers.

Model retained decoding with `T_predecode + r*T_dense` against `r*T_fused`. This predicts a possible reuse boundary only when both measured costs are credible. It is not enough to report that packed weights use fewer bytes.

**Correctness:** exhaust all 16 nibble values; odd/tail packing; scales spanning magnitudes that expose premature rounding; positive/negative activations, cancellation and long K; changed scales/weights across repeated launches. Compare the same decoded-weight contract to an FP64 dot-product oracle on small cases. Global and local scales must not be reassociated in a way that changes the admitted numerical contract without testing and labeling it.

**Falsifier:** bandwidth savings do not survive decode, layout and accumulation costs, or apparent speed depends on a lower-precision path. A negative result should preserve a useful memory-saving option with an honest throughput boundary, not imply compressed storage is impossible on V100.

## Bounded sparse-matrix companion

The CUDA skill's blocked-sparse guidance is useful when real dense blocks exist. A compatible cuSPARSE Blocked-ELL path supplies a baseline for the documented supported type/operation combination; it is not proof that arbitrary FP32-only or integer tensor modes are native to V100. Check the installed API's restrictions. [EXT-CUSPARSE]

Do not launch a new sparse-library project. Characterize one already-available block-sparse case if it helps the grouped/selected-block workload. Include padding, index handling, conversion and reuse. A dense-tile speedup that loses end-to-end is not a successful sparse optimization.

## Publication outcome

A grouped/gated case study and a software low-bit-decoding technique/kernel page, with precise format and compute distinctions. Give shape/reuse-dependent dispatch advice from measurements. Do not market a W4A16 implementation as native NVFP4 or INT4 Tensor Core support, and do not claim complete model/format compatibility.
