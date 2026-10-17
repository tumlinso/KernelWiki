---
id: lang-volta-cuda-cpp
title: "CUDA C++ smoke path for Volta"
type: language
tags: [cuda-cpp, wmma]
related: [lang-volta-ptx, hw-volta-mma]
sources: [doc-volta-cuda-12-9, doc-volta-ptx-8-8]
reproducibility: snippet
architectures: [sm70]
confidence: source-reported
---

# CUDA C++ on Volta

CUDA C++ source can combine host-side CUDA Runtime calls with device functions and kernels. The snippet below is copied from the shared native smoke source, not a free-standing example; the source file is the canonical version.

```cpp
constexpr int kSide = 16;

__global__ void wmma_16x16x16(const half* a, const half* b, float* output) {
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  wmma::load_matrix_sync(a_frag, a, kSide);
  wmma::load_matrix_sync(b_frag, b, kSide);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(output, c_frag, kSide, wmma::mem_row_major);
}
```

This WMMA operation uses FP16 inputs and an FP32 accumulator for a 16×16×16 product. WMMA exposes fragments through its API; this example does not establish a raw `mma.sync` lane/register mapping. The exact kernel is in the [authored native source](../../artifacts/kernels/volta-common/variants/volta_smoke.cu). It compiled with CUDA 12.9.86 for `sm_70` and ran on controller-selected V100 SM70 hardware. The input and numerical policy are recorded in [`benchmark_runs/volta-environment/contract.json`](../../benchmark_runs/volta-environment/contract.json); the native binary checks each output against a binary64 host reference. The archived measured executable, source hash, controller leases, and results are linked by [`benchmark_runs/volta-environment/summary.json`](../../benchmark_runs/volta-environment/summary.json).

CUDA C++ and CUTLASS C++ are both C++ source paths, but CUTLASS adds template/library abstractions and its own versioned APIs. CuTe's Python DSL is a separate Python authoring interface, even when it lowers through the CUDA toolchain. Triton is a separate source language with its own compiler/lowering path. Evidence for one path does not transfer to another.

**Qualification state:** the environment smoke passed for this one WMMA shape. Baseline and candidate labels used the same source and binary, with identical median launch time; this is not an optimization comparison. The sanitizer wrapper retains its raw `partial` classification; qualification is limited to zero-error memcheck and the bound native `PASS` record. The selected NCU report was readable, but source-stall metrics were absent and PM sampling was unsupported. No claim is made for CUTLASS, CuTe DSL, Triton, general GEMM shapes, or raw MMA fragment layouts. See the [measured scope and limits](../../benchmark_runs/volta-environment/README.md).
