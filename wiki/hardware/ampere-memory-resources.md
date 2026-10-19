---
id: hw-ampere-memory-resources
title: 'Ampere SM80: A100 memory and resource limits'
type: hardware
architectures:
- sm80
tags:
- register-budgeting
- shared-memory-optimization
- cache-policy
- cp-async
confidence: source-reported
reproducibility: concept
related:
- hw-ampere-async-copy
- hw-volta-resources
sources:
- doc-ampere-tuning-guide-12-9
- doc-ampere-cuda-12-9
aliases: []
---

These values describe the documented SM80/A100 configuration. Ampere is a family, not one uniform resource profile: compute-capability 8.6/8.7 devices have different shared-memory/cache and warp limits. Query the actual device and compiled kernel before using a number in a launch or performance model.

For SM80 the tuning guide reports 64 warps per SM, a 64K-entry 32-bit register file, up to 255 registers per thread, 164 KB shared memory per SM, and up to 163 KB addressable by a CTA. CUDA documents the combined L1/shared data cache as 192 KB for compute capability 8.0. Shared memory is a configurable carveout of that unified resource; dynamic allocations above 48 KB require explicit opt-in, and the usable allocation is below the whole per-SM partition. A preferred carveout is a request to the driver, not a guarantee that the remaining L1 or residency will match a fixed model.

The A100 tuning guide reports a 40 MB L2 and describes L2 residency controls. Treat these as architecture/product properties, not as proof that a particular working set remains resident or that a cache policy improves a workload. Verify the current SKU, allocation shape, launch, and access stream.

When combining a multi-stage `cp.async` pipeline with Tensor Core work, account for stage buffers, operand/accumulator registers, shared-memory carveout, and resident CTAs together. The max shared-memory value is not a throughput target; more staging can reduce the number of resident blocks or the L1 available to other accesses. No local performance threshold is established here.
