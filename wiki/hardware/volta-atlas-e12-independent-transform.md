---
id: hw-volta-atlas-e12-independent-transform
title: 'Volta experiment E12: independent local transform'
type: hardware
architectures:
- sm70
tags:
- mma-sm70
confidence: source-reported
reproducibility: concept
related:
- hw-volta-mma
- hw-volta-numerics
- hw-volta-atlas-evidence-index
sources:
- blog-atlas-v100-campaign-20261006
aliases: []
---

E12 compared shuffle/add with FP16 WMMA for a block-diagonal transform made
from four independent normalized 4-by-4 Hadamard blocks. Each iteration
recomputed the transform of the same fixed input and accumulated a scaled
result. It did not feed the prior output back as a recurrent state update
`M^r x`, and it did not test a coupled 16-variable matrix.

The selected cases passed their references. WMMA showed larger finite-case
error than shuffle/add under FP16 conversion; these observations provide no
general precision bound. Timings were kernel-resident and omitted full input
preparation, routing, conversion, and extraction costs. The result therefore
does not establish a complete-pipeline advantage for tensor arithmetic.

The pinned [E12 card](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/E12.md)
and [portable E12 summary](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E12.json)
retain the checked cases and metric detail. The Atlas URLs are access-gated;
use the source record's local pinned-Git fallback.
