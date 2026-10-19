---
id: hw-volta-feature-boundaries
title: 'V100: native features versus software representations'
type: hardware
architectures:
- sm70
tags:
- mma-sm70
- ptx
- cuda-cpp
confidence: source-reported
reproducibility: concept
related:
- hw-volta-mma
- hw-volta-numerics
- hw-volta-participation
- hw-volta-vmm-addressing
- hw-volta-integer-routing
- hw-ampere-async-copy
- hw-ampere-tensor-operations
sources:
- doc-volta-ptx-8-8
- doc-volta-capability
- doc-volta-cuda-12-9
aliases: []
---

V100 is the exact SM70 target. A family label or a source page about another target is not an exact V100 capability. In particular, integer matrix modes documented for Turing or later do not establish native V100 Tensor Core support.

Do not import `cp.async` (SM80+), `ldmatrix` (SM75+), TMA, warp-group MMA, TMEM, or cluster/CLC mechanisms as executable SM70 instructions. These later forms have distinct targets and contracts. Rebuild required dataflow with supported loads, registers/shared storage, and synchronization only when the replacement is both correct and worthwhile; do not infer a native feature from a software representation with the same data format.

An application can use public APIs whose support is queried at run time (for example, CUDA virtual memory management); those API capabilities are distinct from instruction eligibility. Driver headers, open-GPU definitions, channel methods, and firmware command fields may expose implementation details, but their presence does not make them supported CUDA application interfaces or authorize ordinary kernels to use them. See [the public VMM contract](volta-vmm-addressing.md).

A packed FP8/FP4/INT4 storage format may be decoded into arithmetic supported by SM70. That does not create native tensor arithmetic for the stored format. Keep storage representation, compute type, API/ISA eligibility, and numerical qualification separate, and retain exact target and toolchain notes. An archived target or compiler's PTX support does not guarantee that a current library, binary, or driver will provide a working native path.
