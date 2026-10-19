---
id: hw-ampere-async-copy
title: 'Ampere SM80: asynchronous global-to-shared copy and split barriers'
type: hardware
architectures:
- sm80
tags:
- cp-async
- split-arrive-wait
- pipeline-stages
- double-buffering
- cuda-cpp
confidence: source-reported
reproducibility: concept
related:
- hw-ampere-memory-resources
- hw-volta-resources
- hw-volta-participation
sources:
- doc-ampere-tuning-guide-12-9
- doc-ampere-cuda-12-9
- doc-ampere-ptx-8-8
aliases: []
---

SM80 adds hardware-accelerated asynchronous global-to-shared copies. They can overlap movement with independent computation and avoid the ordinary global-load-to-register-to-shared-store staging path. PTX offers `.ca` and `.cg` cache policies: `.ca` allows caching at all levels including L1, while `.cg` caches at L2 and bypasses L1. These are cache hints, not memory-ordering guarantees. The CUDA `memcpy_async` and pipeline/barrier APIs are available on earlier targets as software/API mechanisms too; hardware acceleration is the SM80 distinction.

## Eligibility and completion

The PTX `cp.async` family copies global memory to shared memory in 4-, 8-, or 16-byte units. The CUDA API can use an exclusively asynchronous path when pointer alignment and copy size prove the required alignment; otherwise it can select a fallback, sometimes after a runtime check. The programming guide calls for 128-byte pointer alignment for best API performance. Measure generated code and full-kernel time rather than assuming that a function named `memcpy_async` emitted the accelerated form.

The data is not ready for an ordinary shared-memory load until the matching async copy group or API pipeline/barrier has completed. Buffer reuse also needs an ordering point after consumers finish. `cp.async.wait_group` completion, CTA barriers, and pipeline stages solve related but distinct parts of that lifecycle; do not treat issue as completion or a wait as automatic proof that every other consumer has finished.

SM80 also provides hardware acceleration for split arrive/wait barriers in shared memory. A correctly initialized barrier expresses a particular participant count and phase. Its use does not create a grid-wide barrier, guarantee that a producer CTA will be scheduled, or make an ordinary oversubscribed spin protocol progress. Use the exact barrier and pipeline contract from the CUDA guide.

## Cost and scope

The mechanism is most relevant when a regular tile can be prefetched far enough ahead to overlap useful work. Extra stages consume shared memory and may lengthen live ranges, changing occupancy and cache carveout. Irregular tails, tiny tiles, insufficient independent work, or a different bottleneck can erase the benefit. Compare against the simple load/store path and include pipeline setup, synchronization, registers, shared memory, and end-to-end time.

This page is scoped to SM80. It does not import Hopper TMA or warp-group synchronization, and it establishes no local timing result.
