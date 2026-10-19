---
id: hw-volta-atlas-e11-packed-intersection
title: 'Volta experiment E11: packed set intersections'
type: hardware
architectures:
- sm70
tags: []
confidence: source-reported
reproducibility: concept
related:
- hw-volta-mma
- hw-volta-atlas-evidence-index
sources:
- blog-atlas-v100-campaign-20261006
aliases: []
---

E11 compared packed AND plus population count, DP4A byte expansion, and FP16
WMMA for the same dense matrix of exact set-intersection counts. The
correctness-passing cases used a fixed 16-by-16 query matrix at three selected
universe sizes. Packed AND+POPC had the lowest measured complete-path median
for each selected case. That path included representation encoding where
needed, host-to-device transfer, kernel, device-to-host transfer, and result
materialization; common source generation and allocation were outside it.

The configured reuse values did not become repeated-use work: the runner set
the actual iteration count to one for all cases. E11 therefore supplies no
reuse sweep. Nor does the comparison establish superiority over general CSR,
sparse-output contracts, other densities, or other intersection workloads.
FP16 count exactness was observed only for the tested bounded nonnegative
inputs; it is not a numerical guarantee. See the separate [Volta numerical
fingerprint](volta-numerics.md).

The pinned [E11 card](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/E11.md)
and [portable E11 summary](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E11.json)
are the scope and evidence source. The Atlas interpretation guide distinguishes
this set operation from E39 integer addition. The Atlas URLs are access-gated;
use the source record's local pinned-Git fallback.
