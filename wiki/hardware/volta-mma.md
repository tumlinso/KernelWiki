---
id: hw-volta-mma
title: 'Volta: explicit MMA and opaque WMMA'
type: hardware
architectures:
- sm70
tags:
- mma-sm70
- wmma
- gemm
confidence: source-reported
reproducibility: concept
related:
- hw-volta-numerics
- hw-volta-participation
sources:
- doc-volta-ptx-8-8
- doc-volta-tuning-12-9
- blog-volta-atlas-campaign
aliases: []
---

The explicit half-input `mma.sync.aligned.m8n8k4` form and C++ WMMA are distinct interfaces. Use the exact PTX fragment contract for the selected A/B layout and accumulator type; do not infer raw WMMA ownership from that contract. A full participating warp is required even though the explicit Volta form contains four independent small products.

Kernel design should account for the producer accumulator layout, the next consumer's operand layout, and conversions or lane exchanges between them. Accumulator precision can change that movement. A layout shortcut is not a like-for-like optimization when it changes FP32 accumulation to FP16.

The atlas E09 explicit row/column FP32 form is an existing coordinate-validation source, not a general raw-WMMA map. Import its exact code/measurement provenance only after inspecting the bounded evidence; a new binary still needs a short correctness check. This draft reports no new compilation or benchmark.
