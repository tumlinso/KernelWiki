---
id: hw-volta-atlas-e39-bitplane-addition
title: 'Volta experiment E39: bitplane full-adder comparison'
type: hardware
architectures:
- sm70
tags: []
confidence: source-reported
reproducibility: concept
related:
- hw-volta-numerics
- hw-volta-resources
- hw-volta-atlas-evidence-index
sources:
- blog-atlas-v100-campaign-20261006
aliases: []
---

E39 compared native integer arithmetic, a bitplane representation, and a
512-entry nibble lookup for the repeated transition
`total = uint64(x) + uint64(b) + carry`, with the low word becoming `x` and the
high word becoming `carry`. The selected matrix included boundary and larger
sizes, several reuse counts, and random, alternating, and carry-heavy inputs.
All recorded cases passed the CPU oracle. Native arithmetic generally had the
lower median for the selected matched full paths. A small bitplane median
ordering in one cell had an uncertainty interval crossing zero, so the
campaign established no statistical crossover.

The full path included transfers, applicable encoding/decoding, kernel work,
synchronization, and output materialization. Device workspace allocation and
nibble-table setup were outside the measured path; energy was not measured.
The result is specific to this full-adder workload and does not disfavor
bitplanes for Boolean operations such as E11's different set-intersection
operator. It does not establish a broad composition, energy, memory, or
topology result.

The pinned [E39 card](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/E39.md)
and [portable E39 summary](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E39.json)
contain the matched cells, full-path contract, and bounded statistical result.
The Atlas URLs are access-gated; use the source record's local pinned-Git
fallback.
