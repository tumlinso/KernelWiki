---
id: hw-volta-mma
title: 'Volta: explicit MMA and opaque WMMA'
type: hardware
architectures:
- sm70
tags:
- mma-sm70
- wmma
- gemm
confidence: source-reported
reproducibility: concept
related:
- hw-volta-numerics
- hw-volta-participation
- technique-volta-fragment-layout
- hw-volta-atlas-e09-mma-coordinate
- hw-volta-atlas-e11-packed-intersection
sources:
- doc-volta-ptx-8-8
- doc-volta-tuning-12-9
- blog-volta-atlas-campaign
- blog-atlas-v100-campaign-20261006
aliases: []
---

The explicit half-input `mma.sync.aligned.m8n8k4` form and C++ WMMA are distinct interfaces. For the documented FP16-input, FP32-accumulator form, one warp instruction computes four independent 8×8×4 MMA operations. Each operation uses lanes `{4g,…,4g+3}` and `{16+4g,…,19+4g}` for `g = 0..3`; all 32 lanes still execute the same aligned warp collective. Do not branch so only one lane group reaches it.

For this exact row-major A, column-major B, FP32 C/D instruction form, the PTX coordinate map is:

```text
lane l, A fragment element i ∈ [0,3]:
  row = (l mod 4) + 4·[l ≥ 16]
  col = i

lane l, B fragment element i ∈ [0,3]:
  row = i
  col = (l mod 4) + 4·[l ≥ 16]

lane l, accumulator slot i ∈ [0,7]:
  row = (l & 1) + (i & 2) + 4·[l ≥ 16]
  col = (i & 4) + (l & 2) + (i & 1)
```

Each lane supplies four FP16 values for A and B in two packed `.f16x2` registers apiece, and eight FP32 accumulator values. The input and accumulator layouts are separate contracts. Use the PTX fragment table for other major orderings, element types, shapes, or accumulator types; this map does not define a C++ WMMA fragment's raw lane ownership.

The bounded Atlas [E09 coordinate check](../hardware/volta-atlas-e09-mma-coordinate.md) compared this explicit form with a CPU coordinate map across 256 coordinates and reported zero map error for that tested case. Its WMMA case checked canonical output and finite writes, not raw WMMA ownership. It did not establish application-level speed or generalize to other forms. Existing source-level fragment examples are in [the Volta fragment-layout technique](../techniques/volta-fragment-layout.md).

When consuming a fragment in another operation, the exact map can expose producer-consumer ownership matches or mismatches. A register-resident composition may still pay for shuffles, shared-memory exchange, conversion, padding, or extraction. The FP16 accumulator map differs from the FP32 map; changing accumulator type also changes numerical behavior. See [Volta numerics](volta-numerics.md) before treating two layouts as a like-for-like comparison.
