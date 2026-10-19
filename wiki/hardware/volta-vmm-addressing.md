---
id: hw-volta-vmm-addressing
title: 'Volta: public virtual memory mapping and alias contracts'
type: hardware
architectures:
- sm70
tags:
- vmm
- cuda-cpp
confidence: source-reported
reproducibility: concept
related:
- hw-volta-feature-boundaries
- hw-volta-atlas-e23-vmm-aliasing
sources:
- doc-volta-cuda-12-9
- doc-volta-capability
- blog-atlas-v100-campaign-20261006
aliases: []
---

CUDA's public Driver API Virtual Memory Management (VMM) separates virtual-address reservation, physical allocation, mapping, and access rights. The guide's supported sequence uses `cuMemAddressReserve`, `cuMemCreate` (or an imported allocation handle), `cuMemMap`, and `cuMemSetAccess`; the device must support VMM, allocation and mapping alignment must satisfy the queried granularity, and a mapping is not accessible until access is granted. Query `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` and `cuMemGetAllocationGranularity` on the actual device and allocation properties.

Multiple virtual addresses may map the same physical allocation, and allocations from different devices may occupy one contiguous virtual-address range. Neither property makes the physical pages contiguous or uniform in latency, grants access to an otherwise unauthorized device, creates remote shared memory, or combines separate devices into one execution domain. Mapping changes require the range to be unmapped first and must be ordered outside live accesses.

Virtual aliases have an explicit consistency rule. Writes through one proxy are not automatically coherent with other aliases during the same device operation; order alias accesses at operation boundaries, or use the documented `fence.proxy.alias` when the same kernel must access the allocation through distinct aliases. Ordinary pointer equality, `volatile`, or a successful map call is not a substitute for this contract.

The Atlas [E23 result](volta-atlas-e23-vmm-aliasing.md) used fixed aliases over one V100 allocation and separate device walks. For its 4,096- and 65,536-element requests, 2 MiB allocation granularity produced the same 524,288-element logical domain. It measured a difference against a serial modulo ring on that domain, but did not isolate modulo as the cause, test optimized mask/two-segment baselines, establish a TLB mechanism, or test live remapping/concurrent alias coherence. Treat that as a bounded observation, not a VMM performance rule.

This page covers public CUDA VMM. Reverse-engineered page-table edits, driver channel methods, firmware command fields, or physical-address observations are separate restricted interfaces and do not expand the ordinary CUDA contract.
