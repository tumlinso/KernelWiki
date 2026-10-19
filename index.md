# Volta, Ampere, Hopper & Blackwell Kernel Knowledge Base

> Blackwell (SM100) and Hopper (SM90) kernel corpus, source-backed Volta/V100 (SM70) and Ampere/A100 (SM80) hardware contracts, and bounded native V100 kernel case studies.
> Optimized for LLM agent retrieval. See [CLAUDE.md](CLAUDE.md) for schema and conventions.
> **For Claude Code agents**: this repository is a Claude Code skill — see [SKILL.md](SKILL.md).

## Recommended Query Tools (for LLM agents)

```bash
python3 scripts/query.py "<natural language>" [--tag <t>] [--type <kernel|technique|pr|...>]
python3 scripts/get_page.py <page-id-or-path> [--follow-sources]
python3 scripts/grep_wiki.py "<regex>" [--only wiki|sources]
```

See [references/examples.md](references/examples.md) for 10 worked query patterns.

## Quick Navigation

| I want to... | Go to |
|---|---|
| Ask about V100 / Volta (`sm70`) | [Volta routing reference](references/volta.md) |
| Ask about A100 / Ampere (`sm80`) | [Ampere routing reference](references/ampere.md) |
| Browse exact, family-only, or unknown architecture evidence | [queries/by-architecture.md](queries/by-architecture.md) |
| Fix a performance problem | [queries/by-problem.md](queries/by-problem.md) |
| Learn a specific technique | [queries/by-technique.md](queries/by-technique.md) |
| Use a hardware feature | [queries/by-hardware-feature.md](queries/by-hardware-feature.md) |
| See what a repo contributed | [queries/by-repo.md](queries/by-repo.md) |
| Write a specific kernel type | [queries/by-kernel-type.md](queries/by-kernel-type.md) |
| Use a specific language/DSL | [queries/by-language.md](queries/by-language.md) |

## Hardware Features

- [hw-tcgen05-mma](wiki/hardware/tcgen05-mma.md) — Blackwell MMA instruction (replaces wgmma)
- [hw-tmem](wiki/hardware/tmem.md) — Tensor Memory (CTA-visible 128-lane × 512-column view)
- [hw-clc](wiki/hardware/clc.md) — Cluster Launch Control (dynamic tile scheduling)
- [hw-tma](wiki/hardware/tma.md) — Tensor Memory Accelerator (async bulk loads)
- [hw-2sm-cooperative](wiki/hardware/2sm-cooperative.md) — Two-SM cooperative MMA
- [hw-nvfp4](wiki/hardware/nvfp4.md) — NVFP4 and block-scaled narrow precision
- [hw-pdl-gdc](wiki/hardware/pdl-gdc.md) — Programmatic Dependent Launch / Grid Dependency Control
- [hw-ampere-async-copy](wiki/hardware/ampere-async-copy.md) — SM80 asynchronous global-to-shared copy and split barriers
- [hw-ampere-tensor-operations](wiki/hardware/ampere-tensor-operations.md) — SM80 Tensor Core arithmetic eligibility
- [hw-ampere-memory-resources](wiki/hardware/ampere-memory-resources.md) — SM80/A100 cache, shared-memory, and residency limits

## Volta / V100 (SM70) Reference

These hardware pages are source-reported concept guidance. Exact V100 queries use `sm70`; the `volta` family route can also include neighboring `sm72` evidence. Local kernel measurements below are separate experimental results, bounded to their exact contracts and shapes.

- [hw-volta-mma](wiki/hardware/volta-mma.md) — explicit MMA and opaque WMMA boundaries
- [hw-volta-numerics](wiki/hardware/volta-numerics.md) — representation and accumulation contracts
- [hw-volta-resources](wiki/hardware/volta-resources.md) — coupled register, shared-memory, and residency budgets
- [hw-volta-participation](wiki/hardware/volta-participation.md) — collective participation and progress
- [hw-volta-feature-boundaries](wiki/hardware/volta-feature-boundaries.md) — native SM70 features and software representations
- [hw-volta-operand-delivery](wiki/hardware/volta-operand-delivery.md) — profiling-led operand delivery investigation
- [hw-volta-vmm-addressing](wiki/hardware/volta-vmm-addressing.md) — public VMM mapping, permission, and alias consistency contracts

Use [`references/volta.md`](references/volta.md) to route a question and follow each page's source IDs before applying it.
Use [`references/ampere.md`](references/ampere.md) for exact SM80/A100 copy, Tensor Core, and memory-resource retrieval.

## Native V100 Kernel Studies

Each page links its source, runnable artifact, immutable case contract, and
native evidence. These are measured case studies, not universally optimal
implementations or cross-architecture recommendations.

| Family | Page | Tested boundary / result |
|---|---|---|
| GEMM | [kernel-volta-gemm](wiki/kernels/volta-gemm.md) | FP16 input / FP32 accumulation; cuBLAS favored dense and small-M, ragged comparison inconclusive between cuBLAS/PTX while pinned CUTLASS won that tested ragged case |
| Row kernels | [kernel-volta-row-kernels](wiki/kernels/volta-row-kernels.md) | CTA softmax and aligned vector RMSNorm beat matched routes for three tested shapes |
| Attention | [kernel-volta-attention](wiki/kernels/volta-attention.md) | Prefill inconclusive; unsplit SIMT decode lost; 16-way split won one exact D=70 decode shape including merge; selected-block speedup unmeasured |
| Grouped/gated | [kernel-volta-grouped-gated](wiki/kernels/volta-grouped-gated.md) | Paired SIMT won balanced groups; split cuBLAS won a skewed distribution |
| W4A16 | [kernel-volta-w4a16](wiki/kernels/volta-w4a16.md) | Software decode plus dense cuBLAS favored repeated resident batches; no native 4-bit instruction claim |
| Recurrent state | [kernel-volta-gated-delta](wiki/kernels/volta-gated-delta.md) | Retained state favored two prefill cases; one-token decode inconclusive; chunking deferred |

The separate ordered GEMM frontend is documented in
[`artifacts/kernels/volta-gemm-ordered/variants/README.md`](artifacts/kernels/volta-gemm-ordered/variants/README.md)
with [fresh correctness results](benchmark_runs/volta-gemm-ordered/RESULTS.md)
for all six routes on three shapes and both one-hot stage counts. This
separately built frontend has no new timing result.

## Optimization Techniques

- [technique-warp-specialization](wiki/techniques/warp-specialization.md) — Warp role assignment
- [technique-persistent-kernels](wiki/techniques/persistent-kernels.md) — Persistent kernel patterns with CLC
- [technique-swizzling](wiki/techniques/swizzling.md) — Shared memory swizzling
- [technique-pipeline-stages](wiki/techniques/pipeline-stages.md) — Software pipelining
- [technique-epilogue-fusion](wiki/techniques/epilogue-fusion.md) — Fusing epilogue with mainloop
- [technique-tile-scheduling](wiki/techniques/tile-scheduling.md) — Tile scheduling strategies
- [technique-double-buffering](wiki/techniques/double-buffering.md) — Double/multi-buffering
- [technique-software-exp](wiki/techniques/software-exp.md) — Software-emulated exponential
- [technique-fine-grained-quantization](wiki/techniques/fine-grained-quantization.md) - Fine-grained FP8/FP4 quantization
- [technique-vectorized-loads](wiki/techniques/vectorized-loads.md) — Wide vectorized loads and cache policies
- [technique-volta-attention-dataflow](wiki/techniques/volta-attention-dataflow.md) — Online and split attention dataflow with a bounded decode result
- [technique-volta-grouping-and-decoding](wiki/techniques/volta-grouping-and-decoding.md) — Grouped work and W4A16 decode placement

## Kernel Case Studies

- [kernel-flash-attention-4](wiki/kernels/flash-attention-4.md) — FlashAttention-4 (up to 1613 TFLOPS on B200 in the paper's benchmark sweep)
- [kernel-deepgemm](wiki/kernels/deepgemm.md) — DeepGEMM FP8 GEMM (1550 TFLOPS on H800)
- [kernel-flashmla](wiki/kernels/flashmla.md) — FlashMLA sparse/dense MLA decoding
- [kernel-nsa](wiki/kernels/nsa.md) — Native Sparse Attention (9x fwd speedup)
- [kernel-gated-delta-net](wiki/kernels/gated-delta-net.md) — Gated Delta Net linear attention
- [kernel-nvfp4-gemm](wiki/kernels/nvfp4-gemm.md) — NVFP4 GEMM from GPU Mode hackathon
- [kernel-nvfp4-gemv](wiki/kernels/nvfp4-gemv.md) — NVFP4 batched GEMV optimization
- [kernel-grouped-gemm](wiki/kernels/grouped-gemm.md) — Grouped GEMM for MoE
- [kernel-fused-moe](wiki/kernels/fused-moe.md) — Fused MoE with FP8

## Problem → Solution Patterns

- [pattern-low-sm-utilization](wiki/patterns/low-sm-utilization.md) — SM utilization is low
- [pattern-memory-bound](wiki/patterns/memory-bound.md) — Kernel is memory bandwidth limited
- [pattern-register-pressure](wiki/patterns/register-pressure.md) — Too many registers → low occupancy
- [pattern-compute-bound](wiki/patterns/compute-bound.md) — Not reaching peak FLOPS
- [pattern-tail-effect](wiki/patterns/tail-effect.md) — Last wave underutilizes GPU

## Languages & DSLs

- [lang-cute-dsl](wiki/languages/cute-dsl.md) — CuTe DSL for Blackwell
- [lang-cuda-cpp](wiki/languages/cuda-cpp.md) — CUDA C++ with PTX inline
- [lang-ptx](wiki/languages/ptx-sm100.md) — PTX instructions for SM100
- [lang-triton](wiki/languages/triton-blackwell.md) — Triton on Blackwell

## Migration Guides

- [migration-wgmma-to-tcgen05](wiki/migration/wgmma-to-tcgen05.md) — Hopper wgmma → Blackwell tcgen05
- [migration-register-to-tmem](wiki/migration/register-to-tmem.md) — Register accumulators → TMEM

## Source Repositories

| Repository | Focus |
|---|---|
| [NVIDIA/cutlass](queries/by-repo.md#nvidiacutlass) | CUTLASS 4.x Blackwell support |
| [sgl-project/sglang](queries/by-repo.md#sgl-projectsglang) | SGLang Blackwell integration |
| [vllm-project/vllm](queries/by-repo.md#vllm-projectvllm) | vLLM Blackwell support |
| [flashinfer-ai/flashinfer](queries/by-repo.md#flashinfer-aiflashinfer) | FlashInfer Blackwell kernels |
| [pytorch/pytorch](queries/by-repo.md#pytorchpytorch) | PyTorch/Inductor Blackwell |

## Competitions

- [GPU Mode NVFP4 Hackathon](sources/contests/gpu-mode-nvfp4/) — 4 NVFP4 kernel challenges on B200
- [FlashInfer MLSys 2026](sources/contests/flashinfer-mlsys26/) — MoE, Sparse Attention, GatedDeltaNet
