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
sources:
- doc-volta-ptx-8-8
- doc-volta-capability
- doc-volta-cuda-12-9
aliases: []
---

V100 is the exact SM70 target. Volta-family evidence that applies to a different target is not automatically a V100 capability. In particular, do not treat integer matrix modes available on a neighboring architecture as native V100 Tensor Core support.

Do not import `cp.async`, `ldmatrix`, TMA, TMEM, warp-group MMA or cluster/CLC mechanisms as executable SM70 instructions. Rebuild the required dataflow with supported loads, registers/shared storage and synchronization when that is worthwhile.

A packed FP8/FP4/INT4 storage format may be decoded into supported arithmetic. That does not create native tensor arithmetic for the stored format. Keep format decoding, compute type and numerical qualification separate, and retain exact source target notes.
