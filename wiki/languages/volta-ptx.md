---
id: lang-volta-ptx
title: "PTX arithmetic smoke path for Volta"
type: language
tags: [ptx]
related: [lang-volta-cuda-cpp, hw-volta-mma]
sources: [doc-volta-ptx-8-8, doc-volta-cuda-12-9]
reproducibility: snippet
architectures: [sm70]
confidence: source-reported
---

# PTX instructions in the Volta smoke

PTX is a virtual instruction set consumed by NVIDIA's toolchain and lowered for a target GPU. Inline PTX embedded in CUDA C++ remains part of that C++ program's compilation and host/device build path.

The following helper is copied from the shared native smoke source. The kernel calls it for each of 64 input values, and the harness compares the results with a host `std::fma` reference using the same represented inputs.

```cpp
__device__ __forceinline__ float ptx_fma_rn(float a, float b, float c) {
  float result;
  asm volatile("fma.rn.f32 %0, %1, %2, %3;"
               : "=f"(result) : "f"(a), "f"(b), "f"(c));
  return result;
}
```

The corresponding tested call site is `inline_ptx_fma` in the [canonical smoke source](../../artifacts/kernels/volta-common/variants/volta_smoke.cu). It was compiled for `sm_70` and executed on a controller-selected V100; the native result reported zero maximum absolute error for the 64 values, with the captured contract and evidence linked in [`benchmark_runs/volta-environment/summary.json`](../../benchmark_runs/volta-environment/summary.json). This arithmetic instruction example does not qualify an explicit MMA lane map. For SM70 accumulator-coordinate claims, use the separate, directly tested E09 PTX mapping evidence; opaque WMMA fragments are not evidence for that map.

CUDA/CUTLASS C++ and CuTe's Python DSL are distinct authoring and abstraction paths; Triton also has a separate compiler frontend and lowering path. PTX can be a toolchain output for any of them, but that does not make their source-level semantics or interfaces interchangeable.

**Qualification state:** this inline FMA instruction passed the environment smoke's native check. That result is limited to `fma.rn.f32` for these represented values and the recorded tolerance. It says nothing about explicit MMA lane maps, other PTX instructions, CUTLASS, CuTe DSL, or Triton. See the [smoke result and profile limitations](../../benchmark_runs/volta-environment/README.md).
