# Coverage: parity of engineering usefulness

The existing primer is the coverage reference, not a request to clone every named implementation. Hardware-specific examples are translated into **the underlying operation and performance problem**. Unsupported native facilities are documented as unsupported, not simulated under their original name. [KW-PRIMER]

## Coverage map

| Existing topic | Volta treatment | Implementation/material owner |
|---|---|---|
| Tensor instruction, accumulators, operand layouts | Native FP16-input MMA/WMMA, FP32 default accumulation contract, explicit-layout boundaries | M01; KWV-04 |
| TMEM/register accumulators | Registers/shared memory with lifetime and residency tradeoffs; no TMEM claim | M01/M02 |
| TMA, asynchronous stages, `mbarrier` | Software-prefetched register/shared pipeline and supported warp/CTA synchronization; account for registers and barriers | M01; KWV-04 |
| CLC, 2-SM cooperative, cluster scheduling | No native counterpart on V100. Ordinary tiled/batched work first; bounded persistent grouping only when skew/state reuse warrants it | M04; KWV-07 |
| Warp specialization, ping-pong, multistaging | Compare unified warps against justified role/stage choices; never presume modern producer/consumer barriers exist | M01/M04 |
| Vectorization, swizzling, cache policy | Producer-to-consumer layouts, alignment/tails, useful bytes versus requests, L1/shared tradeoff | M02; KWV-05 |
| Register budgeting, occupancy, instruction stalls | Useful ILP and locality versus residency, not maximum occupancy; targeted operand/unroll experiments only after profiling | M01/M06 |
| FlashAttention / online softmax | SM70-native forward attention, FP32 running state, explicit MMA ownership; correct prefill and decode baselines | M03; KWV-06 |
| FlashMLA / sparse MLA / NSA | Reuse attention/gather and masked-block foundations; document layout-specific adaptation and supported operator contract. No automatic full-model support claim | M03; bounded selected-block case |
| FP8 / NVFP4 GEMM and GEMV | Stored-format decoding into a supported compute path. W4A16 representative first; exact FP8/FP4 formats remain separate semantics | M04; KWV-07 |
| Fused MoE, gated dual GEMM, grouped GEMM | Grouped projections and gated epilogue, preserving expert IDs, duplicates and empty groups; no distributed routing subsystem | M04; KWV-07 |
| Gated Delta Net / chunk parallelism | Correct recurrence and state ownership; a bounded chunked alternative only with a proven equivalent algorithm | M05; KWV-08 |
| Reduction, softmax, normalization, exp | Warp/CTA row kernels and measured fusion boundaries; approximate exp only under a declared contract | M02/M03 |
| CUDA C++, PTX, CUTLASS/CuTe | Native CUDA/CUTLASS `Sm70`; distinguish CuTe C++ from Python DSL support and WMMA from explicit PTX | M00; KWV-02/03 |
| Triton / other DSLs | Version- and lowering-specific capability note; compile/execute before asserting a Tensor Core path. No compiler-port project | M00 |
| PR examples, sources, version claims | Select relevant pinned historical and current primary evidence, preserve provenance and limitations | KWV-02 |
| Diagnostic patterns / migration | Small Volta-specific diagnostic routes and porting explanations; related pages are not compatibility proofs | KWV-02/09 |

## Scope of actual kernel work

Five reusable investigations cover the first useful corpus:

1. Native matrix math, fragment ownership, epilogues and software staging.
2. Row reductions/softmax/normalization and memory/fusion tradeoffs.
3. Fused attention: prefill, low-query decode and a selected-block correctness case.
4. Grouped/gated projections and a precisely defined low-bit-storage operator.
5. Recurrent gated-delta state updates and justified short chunks.

These are not five production libraries. Each should share the same small measurement conventions and reuse compatible source implementations. Where a correct upstream implementation already satisfies the contract, adapt and characterize it instead of rewriting it. A material packet may report an existing baseline as the retained implementation after an honest negative experiment.

## Work order and dependency economy

Register the architecture first. In parallel conceptually, source/material adoption can prepare text while the implementation environment is qualified. GEMM/layout and row-kernel work supply the two building blocks for attention. Grouped/quantized and recurrent work consume only the pieces they need, not the entire attention result. Publication gathers the evidence after each unit; it must not become a last-minute bulk documentation task.

The native plan uses one exclusive execution lane for a predictable initial rollout. Its dependency graph permits later independent delegation, but there is no requirement to create multiple worktrees, a new coordinator system, or cross-repository tasks. The implementing agent may use existing bounded delegation once shared changes are integrated.

## Priorities within each investigation

**First:** eliminate whole intermediate tensors, repeated metadata/address work, mathematically redundant computation and mismatched compute paths.

**Second:** adapt tile/warp/fragment ownership and the software pipeline to the observed shape and precision contract.

**Third:** only when counters support it, tune register pressure, operand delivery, unrolling, instruction mix and approximations. Architectural curiosity alone is not a reason to launch a microbenchmark campaign.

This prioritization is a design judgment, not an assertion that one technique always wins. The material briefs specify tests that can reject each candidate.

## What is deliberately not a task

No V200 virtualization, NVLink/NUMA retopology, device aggregation, CUDA driver reverse engineering, copy-engine/firmware experimentation, MPS tuning, full model serving, new optimizer agent, new database, generic autotuner, comprehensive ISA latency atlas, all-architecture plugin system, or wholesale revalidation of the atlas. Numerical multiword tricks, tensor FFT, and new scientific representations stay in the atlas unless a current KernelWiki kernel need actually calls for them.

Attention backward, full training integration, every quantized file format, exact MLA model integration, and a complete block-sparse library are not initial completion gates. Their algorithmic relevance and missing work must be stated honestly. They can follow concrete user workloads rather than inflating this bootstrap.
