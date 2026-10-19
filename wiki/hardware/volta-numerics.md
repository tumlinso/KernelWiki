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
- hw-volta-feature-boundaries
- hw-volta-atlas-e12-independent-transform
- hw-volta-atlas-e39-bitplane-addition
sources:
- doc-volta-fttn
- doc-volta-ptx-8-8
- blog-atlas-v100-campaign-20261006
aliases: []
---

Input conversion, multiplication, accumulation, intermediate rounding and output conversion are separate error sources. FP32 accumulation cannot recover values already lost during FP16 input conversion. The documented MMA instruction defines a particular parallel evaluation; it is not automatically identical to a serial scalar-FMA sequence or to another reduction order.

For each admitted variant, state input representation and conversion, product format, accumulator format, reduction order, intermediate stores, and output conversion. Test the failure modes the operator can reach: cancellation, exponent gaps, subnormal/range behavior where relevant, long contractions, overflow of encoded partial sums, and changed operand order. A finite test maximum is an observation over those inputs, not a general error bound.

Keep arithmetic and ownership experiments separate. In particular, an FP32 accumulator layout and an FP16 accumulator layout are distinct; changing from one to the other is a numerical change as well as a layout change. If reduced precision or an encoded arithmetic domain is itself under study, name it as a separate variant and test the complete operator or state evolution against an independent oracle. The bounded [E12 transform cases](../hardware/volta-atlas-e12-independent-transform.md) cover selected independent local transforms, not coupled state evolution. The [E39 bitplane addition result](../hardware/volta-atlas-e39-bitplane-addition.md) found native arithmetic generally ahead in the tested cells and no statistically established crossover in the marginal case; it does not rule out other workloads or formulations.

This page reports source contracts and test obligations. It contains no local numerical certification or universal accuracy guarantee.
