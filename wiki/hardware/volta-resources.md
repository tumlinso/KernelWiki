---
id: hw-volta-resources
title: 'Volta: resource budgets are coupled'
type: hardware
architectures:
- sm70
tags:
- register-budgeting
- shared-memory-optimization
- double-buffering
confidence: source-reported
reproducibility: concept
related:
- hw-volta-mma
- hw-volta-participation
sources:
- doc-volta-tuning-12-9
- doc-volta-cuda-12-9
aliases: []
---

Registers, shared memory, warp residency and operand latency must be considered together. Query the actual GPU and compiled kernel rather than infer capacity from a different V100 SKU or a theoretical die configuration. More resident warps are useful only when they provide eligible independent work without sacrificing a better layout or excessive intermediate reuse.

A software pipeline can prefetch through registers/shared memory, but its additional live state may reduce residency or spill. Evaluate the complete tile pipeline, not an isolated copy. Shared-memory capacity and L1 use are coupled; changing carveout is a measured implementation choice, not free capacity.

Publish register/shared/spill evidence next to the exact tile and workload. This page supplies qualitative architecture guidance; numerical performance thresholds belong to measured kernel case studies.
