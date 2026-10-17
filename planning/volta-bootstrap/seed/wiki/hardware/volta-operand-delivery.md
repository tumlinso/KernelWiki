---
id: hw-volta-operand-delivery
title: 'Volta: operand delivery is a testable limiter'
type: hardware
architectures:
- sm70
tags:
- register-reuse
- register-budgeting
- loop-unrolling
confidence: source-reported
reproducibility: concept
related:
- hw-volta-resources
sources:
- doc-volta-jia
- doc-volta-tuning-12-9
aliases: []
---

A kernel with good memory locality can still be limited by dependencies and delivery of register operands. Historical Volta microbenchmarks motivate inspecting a specific hot operand stream, but do not establish the register-bank mapping or instruction latency for every current compiler and binary.

Investigate only after profiling a real kernel indicates an unresolved issue/dependency limiter. Compare source-level changes to independent accumulators, operand reuse or modest unrolling while tracking register allocation, spills, instruction count and normal operator time.

Do not infer a bank-conflict cause from a timing change alone. Preserve a measured performance boundary when available evidence does not identify the mechanism. This is not authorization for binary rewriting or a complete instruction microbenchmark campaign.
