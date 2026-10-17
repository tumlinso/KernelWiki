# M03 — Attention as a Volta-native dataflow

## Operator before implementation

For each query row, define scores `s_j = (q · k_j) / sqrt(d) + mask_j`, probabilities `p = softmax(s)` and output `o = sum_j p_j v_j`. Scale **inside** the softmax. SparkAttention's displayed Equation 1 places the scale outside; its Equation 2 also writes a denominator inconsistent with the surrounding online-normalizer description. Treat these as reasons to derive and test the operator independently, not copy equations from a paper as executable truth. [EXT-SPARK, EXT-SOFTMAX]

Start with finite Q/K/V, explicit masked positions, no dropout, and a documented causal-position convention for non-square query/key lengths. A fully masked row returns a defined zero output in this package's proposed contract; the implementation may instead explicitly reject it, but cannot produce accidental NaNs and claim support. Keep input storage, QK accumulation, softmax state, probability conversion, PV accumulation and output rounding separately specified.

Default optimization lane: FP16 Q/K/V, FP32 dot accumulation and running softmax/output state. Rounding softmax probabilities to FP16 for a Tensor Core PV product is an explicit intermediate approximation that must be evaluated against the semantic oracle and matched baseline. FP16 accumulator alternatives are separate variants, never silent substitutions. [AT-E10, EXT-FTTN, EXT-FASTATTN]

## Reusable stable merge

For a block of valid scores maintain `(m, l, u)`, where `m` is the maximum, `l = sum exp(s_j-m)`, and `u = sum exp(s_j-m) v_j` is unnormalized output. Merge blocks a and b:

```
m = max(m_a, m_b)
a = exp(m_a - m)
b = exp(m_b - m)
l = a*l_a + b*l_b
u = a*u_a + b*u_b
output = u/l
```

Handle empty blocks explicitly before evaluating differences of infinities. This identity supplies online prefill, split-KV reduction, and selected-block attention. It also supplies a metamorphic test: changing block boundaries must preserve the operator within the declared floating-point tolerance. A different summation order is not bitwise equivalence by default.

## Candidate sources and what to take from them

SparkAttention and FastAttention provide concrete Volta layout strategies and show why accumulator type changes the movement needed between QK, softmax and PV. Reuse the reasoning and exact supported instruction interfaces. Do not import their performance ratios, approximate arithmetic assumptions or full training/serving environments as local facts. [EXT-SPARK, EXT-FASTATTN]

The current 1Cat-vLLM code is a valuable implementation candidate for wider grouped-query work and reduced KV address/load overhead. PR 666 is merged at `949728e891aaeb43285e6730aa8b7abc8b986acc`; its metadata explicitly preserves precision and unresolved quality/performance limitations. Inspect the relevant kernel and tests at a pinned revision before extraction, including subsequent fixes when appropriate. This package reviewed the PR description, **not every one of its 51 changed files**. Do not adopt it as already-qualified code. [CAND-1CAT-666]

A compatible existing CUDA/CUTLASS attention implementation may be a cheaper starting point than writing from scratch. Verify actual SM70 dispatch; an upstream project name containing “FlashAttention” is not a support guarantee.

## Experiment A1 — Prefill without score materialization

**Baseline:** independent QK + stable softmax + PV on the same represented inputs, plus any already-correct SM70 fused implementation. Small FP64 CPU calculations establish semantic correctness; the GPU baseline supplies realistic timing.

**Candidate:** fuse blockwise QK/softmax/PV using the M01 ownership mapping. Compare canonical shared exchange with the validated warp transform while preserving FP32 running state. Tune only the tile/stage choices implicated by the profile. Avoid a full score matrix except in the small reference.

**Starting cases:** head dimensions 64 and 128; one ragged dimension supported by a fallback; medium and long sequence lengths; a small number of heads versus enough heads/query tiles to occupy the device. Bound memory use to actual available capacity. Include causal and noncausal masks. These are shape regimes, not a required exhaustive sweep.

**Observe:** full operator time, workspace and intermediate bytes, Tensor Core activity where supported, conversion/shuffle/shared instruction contribution, registers/spills and score/softmax/PV stage balance. If the full score materialization is removed, distinguish that algorithmic gain from a later layout gain.

**Falsifier:** the purported improvement only holds with lower precision, omits work on a mask boundary, or requires packing excluded from its published time. Retain the correct baseline while recording the rejected specialization.

## Experiment A2 — Decode: create useful parallelism, not padded FLOPs

A single/few-query decode has little query-axis work. First compare a warp/SIMT reduction path and a Tensor Core path at the actual small batch/head shape; Tensor Cores are not automatically best when the useful tile is mostly padding.

Two measured alternatives are useful:

**Grouped-query packing:** heads sharing the same K/V block can occupy distinct rows of a wider Q tile, increasing useful matrix width and K/V reuse. Their row normalizers, masks and outputs remain independent. Do not merge logits across heads, duplicate the entire cache, or specialize to a fixed head count. Time packing/unpacking as part of the operator unless an upstream producer genuinely retains that layout.

**Split-KV:** independent key-range partitions produce `(m,l,u)` partials and a stable merge. Compare enough partitions to relieve low parallelism with the unsplit path, including partial-state traffic and the merge kernel. Do not fix the number of partitions to a historically lucky value. Reuse the same merge identity and numerical tests as prefill.

For wide head dimensions, partitioning value/output work can reduce per-warp live state. QK probability work should be shared or its recomputation explicitly costed; splitting dimension D is not permission to omit cross-dimension dot-product contributions.

**Observe:** useful/executed math, number of independently scheduled tiles, KV traffic, address/request work, output merge cost, and full latency. A wider aligned KV load may reduce requests and instruction dependencies while leaving DRAM bytes nearly unchanged. Distinguish those explanations.

**Falsifier:** packing or partial merge costs consume the gain, extra work rises faster than useful utilization, or the result changes with head order. Keep shape-based dispatch boundaries and a robust fallback instead of one universal kernel.

## Experiment A3 — Sparse selection is an operator contract

Add one small, supplied-block selection case to the attention harness. The selected block IDs and per-row masks are **inputs**, not a learned/top-k selector implemented here. Compare against the identical dense masked reference. Preserve per-query membership, causal boundaries, empty sets and final partial blocks.

When neighboring query rows share selected K/V blocks, consider loading a union once while retaining exact membership masks for each row. Reordering blocks must not change membership; duplicated IDs must be either rejected or interpreted consistently with the declared input contract, never silently double-counted. Time any union/planning work. This tests a broadly useful substrate for sparse attention/MLA without building a model-specific serving system.

**Falsifier:** reduced loads are offset by mask/planner work at low overlap, or grouping changes which keys a row attends to. Record overlap/reuse as the applicability variable. Do not attribute native 2:4 sparse Tensor Core support to V100.

## Numerical tests that matter

Use one-hot/constant inputs, mixed-sign values, large finite score contrasts, nearly tied logits, long reductions, alternating input buffers, different head order, and changed block partitioning. Compare per-output error and softmax normalizers, not only average error or a finiteness test. Check mask alignment when query length differs from key length. Test moderate and long sequences separately; short-sequence agreement does not establish accumulation behavior at long context.

Approximate exp or FP16 accumulation is considered only after the retained FP32-state path is correct and profiling shows a meaningful scalar bottleneck. Give it a separate admission/tolerance policy; do not relax the default operation. A full model quality claim requires additional model-specific testing and is outside this package.

## Publication outcome

One Volta attention case study, a stable-merge/fragment-layout technique reference, and clear shape/precision/sparsity limitations. Forward prefill and decode are the mandatory representative kernels. Backward and model integration are explicit follow-on work, not hidden incompleteness in a claim of full training support.
