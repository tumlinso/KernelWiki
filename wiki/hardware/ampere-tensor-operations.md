---
id: hw-ampere-tensor-operations
title: 'Ampere SM80: Tensor Core arithmetic eligibility'
type: hardware
architectures:
- sm80
tags:
- mma
- tf32
- gemm
- cuda-cpp
- ptx
confidence: source-reported
reproducibility: concept
related:
- hw-ampere-async-copy
- hw-volta-mma
- hw-volta-numerics
sources:
- doc-ampere-tuning-guide-12-9
- doc-ampere-cuda-12-9
- doc-ampere-ptx-8-8
aliases: []
---

Compute capability 8.0 adds third-generation Tensor Core forms. The documented SM80 matrix path includes FP16 inputs with FP16 or FP32 accumulation, BF16 inputs with FP32 accumulation, TensorFloat-32 inputs with FP32 accumulation, signed/unsigned 8-bit integer inputs with INT32 accumulation, sub-byte 4-bit integer inputs with INT32 accumulation, binary matrix forms, and FP64 matrix arithmetic. Eligibility depends on the instruction/API form, input type, accumulator type, matrix shape, and target; the presence of a C++ type or a PTX mnemonic alone does not establish a native path.

The Ampere tuning guide lists representative SM80 shapes: FP16/BF16 HMMA uses 16×8×8 or 16×8×16, TF32 HMMA uses 16×8×4, signed/unsigned 8-bit IMMA uses 8×8×16, 16×8×16, or 16×8×32, and signed/unsigned 4-bit IMMA uses 8×8×32, 16×8×32, or 16×8×64. These are instruction-family entries, not a promise that every public API exposes every form or that every compiler emits it for a given source expression.

TF32 preserves the FP32 exponent range while using a reduced-precision matrix operand representation; the accumulator remains FP32. FP16 and BF16 inputs also lose information before accumulation, and changing accumulation order changes the result. Select a numerical mode only against the application's error contract and test the full operator against an independent reference. See [Volta numerical contracts](volta-numerics.md) for the general separation of input conversion, product, accumulation, and output error.

SM80 also supports `mma.sp` structured-sparse forms. For FP16/BF16 and signed/unsigned 8-bit inputs, the 2:4 form encodes two selected positions per group of four in sparse A along K; the unselected positions are structurally zero and the selected positions are identified by the metadata operand. A selected payload value may itself be numerically zero, so an encoded group can have fewer than two mathematical nonzeros. A group with more than two mathematical nonzeros cannot be represented losslessly by this format. Sparse TF32 instead selects one position from each pair, with the unselected position structurally zero; its selected value may also be zero. The instruction consumes this encoded sparse representation; it does not discover or compress arbitrary zeros in a dense input. The sparse instruction is a warp collective, and all lanes must execute it with a valid, consistent instruction and metadata layout. Use the exact type, shape, ordering, and metadata rules in the [SM80 PTX source](../../sources/docs/ampere-ptx-8-8.md). Eligibility is not evidence of a speedup; include metadata generation, sparse layout conversion, and the dense-library baseline in any evaluation. No local performance result is established here.

Use the target-specific PTX/CUDA documentation and compiler output to establish the exact form for a custom path. For ordinary dense GEMM, compare with cuBLAS/cuBLASLt or CUTLASS before owning a lower-level implementation. This page documents capability and eligibility only; it makes no performance recommendation or local measurement claim.
