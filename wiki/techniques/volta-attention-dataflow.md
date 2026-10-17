---
id: technique-volta-attention-dataflow
title: Online and Split Attention Dataflow on Volta
type: technique
architectures: [sm70]
tags: [attention, sparse-attention, conditional-rescaling, software-exp]
confidence: experimental
reproducibility: benchmarked
prerequisites: []
related: [kernel-volta-attention, technique-volta-memory-fusion]
sources: [blog-volta-attention-measurements]
blackwell_relevance: Stable online reductions and explicit partial-state merges provide design patterns to test on newer architectures; this bounded Volta result establishes no cross-architecture performance conclusion or split-selection threshold.
---

# Keep score scaling and normalization in one contract

Attention computes

```text
z[h,i,j] = (q[h,i] dot k[g(h),j]) / sqrt(D)
p[h,i,:] = stable_softmax(mask(z[h,i,:]))
o[h,i,:] = sum_j p[h,i,j] * v[g(h),j,:]
```

Apply `1/sqrt(D)` before the stable softmax. In grouped-query attention,
`g(h)=h//(Hq/Hkv)`, but each query head still has an independent row maximum,
denominator, and value accumulator. Reusing the KV head must not merge query
head normalizers.

The attention artifact uses represented FP16 Q/K/V and FP32 dot accumulation,
probabilities, output, and online state. The online update maintains a maximum
`m`, a normalizer `l`, and an unnormalized value vector `u`. When a block of
scores has a new maximum, rescale the old `l` and `u` by the exponential of
the maximum difference before adding the new block's contribution. The final
output is `u/l` when `l>0`; an empty/all-masked row returns zero.

The essential online rescaling step is visible in this abbreviated CUDA
excerpt. The full kernel maintains one value accumulator per output feature
and applies masks before this update:

```cuda-cpp
const float next_m = fmaxf(running_m, score);
const float old_scale = running_l == 0.0f ? 0.0f : expf(running_m - next_m);
const float new_weight = expf(score - next_m);
running_l = old_scale * running_l + new_weight;
running_u = old_scale * running_u + new_weight * value;
running_m = next_m;
```

# Define the mask before selecting a kernel

For non-square causal attention, the tested convention is suffix alignment:
query row `i` corresponds to absolute key position `Sk-Sq+i`. It requires
`Sq<=Sk`; it is not the alternative top-left alignment. Byte mask value `1`
excludes a score. Causal, boolean, and supplied block selection all contribute
to the active set.

The selected-block contract defines membership only. Indices are unique and
in range, order is immaterial, the tail block is clipped, and an empty list
has zero output. Duplicate indices fail validation. This is not a block
planner or a claim about a sparse routing policy. The measured selected-block
case establishes semantics and correctness, not performance.

# Materialize or stream based on the full operator

A materialized implementation can use library matrix products for QK and PV
with an explicit FP32 softmax between them. Its accounting must include layout
conversion or expansion work needed by the PV product. In the measured
baseline, represented half V is expanded to FP32 and the FP32 probability
matrix multiplies it with SGEMM. Expansion, QK, softmax, and PV all fall inside
the CUDA-event interval; input transfer and allocation do not.

An online SIMT path avoids materializing the full score and probability
matrices by retaining `(m,l,u)` state. That reduction in intermediate storage
does not guarantee lower latency. The runner still reserves score, probability,
and expanded-V buffers in its shared case context for online and split routes;
they remain unused by those operators. No harness peak-memory or
allocation-saving measurement was made. For the measured `Q=(1,8,1,70)`,
`KV=(1,2,4096,70)` decode, its single CTA per query row exposes only eight
CTAs on an 80-SM device and measured 704.512 us versus 148.480 us for the
materialized baseline. The one-query-row grid provides concrete evidence to
consider splitting the key range; the one-sample NCU counters are diagnostic
context, not timing proof.

A split implementation must include its partial reduction and merge in the
same timing boundary. The measured 16-way split used a partial `(m,l,u)` per
split and merged those partials with maximum rescaling; both launches were
timed. It measured 66.560 us for this exact decode shape. This result justifies
the bounded route for this case only. It does not imply that 16 is a universal
split count or that all decode shapes benefit. The prefill comparison
`97.280 us` materialized versus `96.256 us` online was not decisive.

# Volta evidence boundary

The complete native source, fixed contract, CPU FP64 oracle, fixtures, role
receipts, and v1/v2 source closures are documented in
[`volta-attention`](../kernels/volta-attention.md) and
[`volta-attention-measurements`](../../sources/blogs/volta-attention-measurements.md).
The tested domain is bounded to the declared shapes, tolerances,
`D<=128`, and `Sk<=4096`; one separate high-contrast fixture is outside the
default input range. No Tensor Core fused path, 2:4 sparsity, general sparse
planner, or cross-architecture performance claim is made.
