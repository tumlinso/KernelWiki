---
id: hw-volta-atlas-e09-mma-coordinate
title: 'Volta experiment E09: explicit MMA coordinates'
type: hardware
architectures:
- sm70
tags:
- mma-sm70
- wmma
confidence: source-reported
reproducibility: concept
related:
- hw-volta-mma
- hw-volta-participation
- hw-volta-atlas-evidence-index
sources:
- blog-atlas-v100-campaign-20261006
aliases: []
---

E09 ran two small GPU checks on a Tesla V100. The explicit PTX case used
`mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`, saved per-lane accumulator
coordinates before canonicalization, and matched the documented map across
the checked output coordinates with zero CPU-map error. This is evidence for
that documented instruction form, operand layout, compiler, and tested
mapping; it is not a general SASS contract across toolchains.

The separate WMMA case matched canonical output against an FP16-input oracle
and wrote finite raw fragment slots. It did not check which lane owns a slot.
WMMA fragment ownership therefore remains opaque. Keep this finding attached to
the distinction and full-warp contract in [Volta explicit MMA and WMMA](volta-mma.md)
and [participation](volta-participation.md).

The pinned [E09 card](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/E09.md)
and [portable E09 summary](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E09.json)
contain the cases and boundary. The campaign source record notes that raw
vectors and profiler captures are not present in the tracked source tree. The
Atlas URLs are access-gated; use the source record's local pinned-Git fallback.
