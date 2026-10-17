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
sources:
- doc-volta-cuda-12-9
- doc-volta-ptx-8-8
aliases: []
---

Independent thread scheduling does not remove collective participation requirements. A shuffle source must participate and hold the intended value; synchronization masks are contracts, not an instruction to fetch inactive lanes. CTA barriers likewise require the appropriate participants and valid shared-memory ordering.

A stress test cannot establish a synchronization guarantee absent from the API. Design finite work and a progress argument before using resident producer/consumer roles. Do not rely on all ordinary blocks being simultaneously resident or on an oversubscribed global spin barrier.

Keep ordering, participation and progress as separate obligations in the code and tests. Volta software staging must use supported synchronization rather than imported TMA/mbarrier/cluster assumptions.
