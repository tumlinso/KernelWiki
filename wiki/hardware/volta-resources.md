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
- hw-volta-operand-delivery
- hw-volta-atlas-e16-resident-scheduling
sources:
- doc-volta-tuning-12-9
- doc-volta-cuda-12-9
- blog-atlas-v100-campaign-20261006
aliases: []
---

Registers, shared memory, L1 capacity, warp residency, and operand latency form a coupled budget. On the documented Volta configuration, shared memory is carved from the unified L1/shared resource; requesting more shared memory can leave less L1. The supported carveout values and maximum per-block allocation depend on the target. Query the actual GPU and compiled kernel rather than infer enabled capacity from a different SKU or a theoretical die configuration.

A software pipeline can prefetch through registers and shared memory, but additional buffers and longer-lived operands may reduce residency or spill. Evaluate the complete tile schedule, not an isolated copy. More resident warps help only when they supply eligible independent work; they do not compensate automatically for extra instructions, conversions, or a worse ownership layout. Volta does not have the Ampere global-to-shared `cp.async` transfer path.

Publish register/shared/spill evidence next to the exact kernel, tile, compiler, and workload. This page supplies qualitative constraints; occupancy limits are not throughput predictions, and performance thresholds belong to scoped kernel measurements.
