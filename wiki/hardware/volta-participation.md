---
id: hw-volta-participation
title: 'Volta: participation, visibility and progress'
type: hardware
architectures:
- sm70
tags:
- warp-collectives
- cuda-cpp
confidence: source-reported
reproducibility: concept
related:
- hw-volta-mma
- hw-volta-integer-routing
- hw-volta-resources
- hw-volta-atlas-e16-resident-scheduling
sources:
- doc-volta-cuda-12-9
- doc-volta-ptx-8-8
aliases: []
---

Independent Thread Scheduling does not remove collective participation requirements. A shuffle source must participate and hold the intended value; a mask names lanes expected to participate and does not fetch from an inactive or absent lane. PTX `.sync .aligned` warp collectives such as the Volta `mma.sync.aligned.m8n8k4` require all live warp threads to execute the same instruction with compatible operands. A per-computation lane subset in the fragment table is not permission to execute the instruction from a partial warp.

For CTA communication, prove that the actual barrier participants, arrival count, shared-memory writes, and reuse phase agree. A block barrier is not a grid barrier. A stress test cannot establish a synchronization guarantee absent from the API.

Keep ordering, participation, visibility, and progress as separate obligations. Atomicity does not by itself publish earlier payload stores; use the documented memory order and scope for the selected allocation and peer path. A resident producer-consumer design also needs a schedulability argument: do not assume ordinary blocks all become resident together or rely on an oversubscribed grid-wide spin barrier to make progress. [Atlas E16](../hardware/volta-atlas-e16-resident-scheduling.md) found its resident role arrangement slower than batching and graph replay for the tested task stream; this is a workload result, not a proof that persistent kernels are generally slower.

On SM70, staging is software-managed. Use synchronization supported by the selected CUDA/toolchain contract; do not import Ampere `cp.async`/split-barrier behavior or later TMA, warp-group, or cluster synchronization assumptions.
