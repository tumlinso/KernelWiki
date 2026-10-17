---
id: hw-volta-numerics
title: 'Volta: representation and accumulation contracts'
type: hardware
architectures:
- sm70
tags:
- gemm
- attention
confidence: source-reported
reproducibility: concept
related:
- hw-volta-mma
sources:
- doc-volta-fttn
- doc-volta-ptx-8-8
aliases: []
---

Input conversion, multiplication, accumulation, intermediate rounding and output conversion are separate error sources. FP32 accumulation cannot recover values already lost during FP16 input conversion. A Tensor Core result is not automatically the same as a serial scalar-FMA evaluation.

Use feature-targeted tests for cancellation, exponent gaps, subnormal/range behavior where relevant, long contractions and changed operand order. Historical numerical observations and finite test maxima are not universal hardware guarantees or global error bounds.

For a layout experiment, keep the arithmetic contract fixed. If reduced precision is itself being explored, name it as a separate admitted variant and test the complete operator or state evolution. This draft contains no local numerical certification.
