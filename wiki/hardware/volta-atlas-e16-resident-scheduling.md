---
id: hw-volta-atlas-e16-resident-scheduling
title: 'Volta experiment E16: resident scheduling comparison'
type: hardware
architectures:
- sm70
tags: []
confidence: source-reported
reproducibility: concept
related:
- hw-volta-resources
- hw-volta-participation
- hw-volta-atlas-evidence-index
sources:
- blog-atlas-v100-campaign-20261006
aliases: []
---

E16 compared a finite cooperative-resident task loop, ordinary batched
launches, and graph replay for two selected regular sum workloads. All variants
produced the expected sums. Resident execution had higher measured latency
than both batching and graph replay at those two tested sizes.

This result is a bounded negative for the selected task stream, not a finding
that persistent execution is generally slower. The tested work did not sweep
task skew, starvation, expensive retained state, or broad residency/resource
conditions. No fairness or starvation-freedom claim follows from completion of
the finite loop.

The pinned [E16 card](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/E16.md)
and [portable E16 summary](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E16.json)
preserve the finite task counts, rounds, and measurement details. The Atlas
URLs are access-gated; use the source record's local pinned-Git fallback.
