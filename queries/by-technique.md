# Query: By Technique

> Auto-generated. Do not edit manually.

| Technique | Tags | Architectures | Confidence | Reproducibility | Sources |
|-----------|------|--------------|------------|-----------------|---------|
| [CCCL CUB SM100 Scan Tuning](../wiki/techniques/cccl-memory-primitives.md) | cuda-cpp, parallel-scan, vectorized-loads, tile-scheduling | sm100 | source-reported | snippet | 1 |
| [Chunk-Based Parallelism for Linear Recurrent Models](../wiki/techniques/chunk-parallelism.md) | chunk-parallelism, linear-attention, triton | sm90 | source-reported | snippet | 2 |
| [Double/Multi-Buffering Patterns](../wiki/techniques/double-buffering.md) | double-buffering, tmem, pipeline-stages | sm100, sm90 | source-reported | snippet | 3 |
| [Epilogue fusion](../wiki/techniques/epilogue-fusion.md) | epilogue-fusion, tmem, warp-specialization | sm100, sm90 | source-reported | snippet | 2 |
| [External Source-Map Research For Kernel Edits](../wiki/techniques/external-source-map-research.md) | cuda-cpp, cute-dsl, tma, wgmma | sm100, sm90 | source-reported | snippet | 5 |
| [Fine-grained FP8/FP4 scaling](../wiki/techniques/fine-grained-quantization.md) | fine-grained-quantization, fp8, fp4, nvfp4 | sm100, sm90 | source-reported | snippet | 3 |
| [Grouping Work and Decoding Packed Weights on Volta](../wiki/techniques/volta-grouping-and-decoding.md) | tile-scheduling, data-reuse, kernel-fusion, fine-grained-quantization | sm70 | experimental | benchmarked | 1 |
| [Kernel fusion](../wiki/techniques/kernel-fusion.md) | kernel-fusion, fused-kernel, tmem | sm100, sm90 | source-reported | snippet | 4 |
| [Memory Mapping and Fusion for Volta Row Operators](../wiki/techniques/volta-memory-fusion.md) | vectorized-loads, kernel-fusion, shared-memory-optimization | sm70 | experimental | benchmarked | 1 |
| [Online and Split Attention Dataflow on Volta](../wiki/techniques/volta-attention-dataflow.md) | attention, sparse-attention, conditional-rescaling, software-exp | sm70 | experimental | benchmarked | 1 |
| [PTX Cache Policy Differentiation](../wiki/techniques/cache-policy.md) | cache-policy, vectorized-loads | sm100, sm90 | source-reported | snippet | 4 |
| [Persistent Kernels with Cluster Launch Control](../wiki/techniques/persistent-kernels.md) | persistent-kernel, clc, tile-scheduling | sm100 | source-reported | snippet | 4 |
| [Ping-Pong Scheduling](../wiki/techniques/ping-pong-scheduling.md) | ping-pong-scheduling, warp-specialization, tmem, pipeline-stages | sm100 | source-reported | snippet | 2 |
| [Register budgeting](../wiki/techniques/register-budgeting.md) | register-budgeting, register-reuse | sm100, sm90 | source-reported | snippet | 3 |
| [Shared Memory Swizzling](../wiki/techniques/swizzling.md) | swizzling, shared-memory-optimization, tma | sm100, sm90 | source-reported | snippet | 3 |
| [Software Pipelining and Multi-Stage Buffering](../wiki/techniques/pipeline-stages.md) | pipeline-stages, double-buffering, tma, mbarrier | sm100, sm90 | source-reported | snippet | 4 |
| [Software-Emulated Exponential](../wiki/techniques/software-exp.md) | software-exp, attention | sm100 | source-reported | snippet | 2 |
| [Tile Scheduling Strategies](../wiki/techniques/tile-scheduling.md) | tile-scheduling, clc, persistent-kernel | sm100, sm90 | source-reported | snippet | 3 |
| [Volta GEMM software staging and stage-count tradeoffs](../wiki/techniques/volta-software-staging.md) | mma-sm70, gemm, pipeline-stages | sm70 | experimental | benchmarked | 1 |
| [Volta explicit MMA fragment ownership and pointwise consumers](../wiki/techniques/volta-fragment-layout.md) | mma-sm70, gemm, kernel-fusion | sm70 | experimental | benchmarked | 2 |
| [Volta recurrent-state ownership across tokens](../wiki/techniques/volta-state-ownership.md) | kernel-fusion, gated-delta-net | sm70 | experimental | benchmarked | 3 |
| [Warp Specialization on Hopper and Blackwell](../wiki/techniques/warp-specialization.md) | warp-specialization, tcgen05, tmem | sm100, sm90 | source-reported | snippet | 4 |
| [Wide Vectorized Loads and Cache Policies](../wiki/techniques/vectorized-loads.md) | vectorized-loads, cache-policy, register-budgeting | sm100, sm90 | source-reported | snippet | 4 |
