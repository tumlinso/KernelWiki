---
id: hw-volta-integer-routing
title: 'Volta: integer logic, packed selection, and warp routing contracts'
type: hardware
architectures:
- sm70
tags:
- lop3
- prmt
- fns
- dp4a
- warp-collectives
- ptx
- cuda-cpp
confidence: source-reported
reproducibility: concept
related:
- hw-volta-participation
- hw-volta-feature-boundaries
- hw-volta-mma
sources:
- doc-volta-ptx-8-8
- doc-volta-cuda-12-9
- doc-volta-math-12-9
aliases: []
---

## Per-bit Boolean logic

PTX `lop3.b32` evaluates a three-input Boolean truth table independently at each of 32 bit positions. Its 8-bit `immLut` chooses one Boolean function for the instruction; it does not choose a different function per bit. The base instruction targets `sm_50` and later, so it is eligible on SM70. The optional `.and`/`.or` predicate-result qualifier is a separate PTX feature: it is available from PTX ISA 8.2 and targets `sm_70` or later. An accepted PTX form does not by itself establish how source code lowers or its cost.

## Byte permutation and set-bit selection

PTX `prmt.b32` selects four bytes from the concatenated two-register input `{b,a}`. In its generic form each destination byte has a 4-bit selector: the low three bits select one of eight source bytes, and the high bit requests sign replication rather than a literal byte copy. This instruction targets `sm_20` and later. CUDA `__byte_perm(x,y,s)` has its own documented selector contract: it uses four 3-bit indices at selector bit offsets 0, 4, 8, and 12 to select from the eight bytes of `x` and `y`; the upper selector bits are ignored. Do not assume that the CUDA intrinsic exposes PTX generic-form sign replication or that the PTX mode-specific selectors are interchangeable with either form.

PTX `fns.b32` searches a 32-bit mask for a set-bit position relative to a base position and signed offset. `base` must be in `[0,31]`; otherwise behavior is undefined. Offset zero tests the base bit itself, positive and negative offsets search toward higher and lower bit positions, and a missing result is `0xffffffff`. The CUDA `__fns` intrinsic documents the same bounded 32-bit result and sentinel. This is a specific directional set-bit search, not a general prefix-rank operation; callers must handle the sentinel and define behavior at word boundaries.

## Packed byte dot product

PTX `dp4a` forms four pairwise products from two packed 32-bit words and adds them to a 32-bit accumulator. Each operand type independently selects signed-byte or unsigned-byte extension (`.s32` or `.u32`); the accumulator is unsigned only when both inputs are unsigned, and signed otherwise. PTX requires `sm_61` or later, which includes SM70. Bound the accumulated result for the application's value range and use the matching signedness. This is a packed scalar integer instruction, not integer Tensor Core MMA: the documented Volta `mma.sync.aligned.m8n8k4` contract here is FP16 inputs with FP32 accumulators. Do not infer a Volta integer Tensor Core path from packed bytes or from later-architecture integer MMA.

The related PTX `dp2a` form also requires `sm_61` or later. It selects the low or high pair of bytes from one packed 32-bit operand and multiplies them by two packed 16-bit values from the other, accumulating into a 32-bit result with the same signedness rule. The `.lo`/`.hi` choice and input extension are part of the instruction contract; it does not return the two products separately.

## Warp communication and membership

The synchronized ballot and shuffle families operate within a warp and require explicit participating-lane masks. Each caller must be named by its mask, and each non-exited lane named in the mask must execute the same synchronized operation with the same mask. A shuffle source must be a valid participating lane when its value is consumed. These operations communicate register values or predicates; they do not establish a memory-ordering fence. For the full independent-thread-scheduling, visibility, and progress contract, see [Volta participation](volta-participation.md).

`match.any.sync` and `match.all.sync` were introduced for `sm_70` and later. They compare values among lanes in the supplied member mask within one warp; they do not aggregate across a CTA. Their CUDA counterparts `__match_any_sync` and `__match_all_sync` have the same mask participation obligation and do not imply a memory barrier. Do not import Ampere warp reduction instructions such as `redux` into an SM70 path.

These are instruction/API capabilities, not a design or performance recommendation. The strongest baseline depends on the work: use ordinary integer arithmetic, a dense lookup, or a standard library routine when packing, selector construction, lane exchange, or edge handling costs more than the operation saved. Inspect the generated target code and validate masks, shifts, sentinels, signedness, accumulator range, and partial warps before relying on an unusual composition. No local performance result is established here.
