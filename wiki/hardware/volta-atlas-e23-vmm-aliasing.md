---
id: hw-volta-atlas-e23-vmm-aliasing
title: 'Volta experiment E23: fixed virtual-memory alias view'
type: hardware
architectures:
- sm70
tags: []
confidence: source-reported
reproducibility: concept
related:
- hw-volta-resources
- hw-volta-vmm-addressing
- hw-volta-atlas-evidence-index
sources:
- blog-atlas-v100-campaign-20261006
aliases: []
---

E23 compared a fixed virtual-address alias walk with a selected serial modulo
ring for matched logical work and access count. The fixed-alias path had lower
observed device-walk time in this workload. The two requested ring sizes both
rounded to the same 2 MiB allocation domain and the same mapped element count,
so the planned view-size sweep did not occur. Setup time was recorded once per
case, not as a repeated-sample distribution.

The comparison did not include a power-of-two mask or optimized two-segment
ring baseline. It therefore does not isolate modulo as the cause and does not
establish a general VMM, TLB, translation, dynamic-remapping, alias-coherence,
or random-access advantage. The authoritative public mapping/access contract
remains in the [CUDA VMM documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__VA.html).

See the pinned [E23 card](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/E23.md)
and [portable E23 summary](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E23.json).
The Atlas URLs are access-gated; use the source record's local pinned-Git
fallback.
