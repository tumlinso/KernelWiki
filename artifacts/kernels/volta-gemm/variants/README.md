# KWV-04 Volta GEMM sources

This bundle implements the native SM70 GEMM investigation documented in
`wiki/kernels/volta-gemm.md`. All paths use row-major A[M,K], B[K,N], and
FP32 output C[M,N], with FP16 represented inputs, FP32 accumulation, alpha=1,
and beta=0. The three deterministic shape contracts and exact represented
inputs live under `benchmark_runs/volta-gemm/cases/`.

`gemm_contract.hpp` defines the serialized input and reference formats,
FP64 reference generation over the stored binary16 bit patterns, and the
predeclared gamma-scaled FP32 admission tolerance. Because PTX MMA accumulation
order and rounding are unspecified, this threshold is not a universal proved
hardware error bound. `gemm_cpu_oracle.cpp` generates input files and
the immutable per-case FP64 reference before any device run. The native
correctness path consumes that reference file and writes its own measured
per-output errors to controller-captured stdout.

`gemm_baselines.cu/.cuh` provide matched cuBLAS and CUTLASS 2.11.0 Sm70
baselines plus an unfused FP32 bias/ReLU consumer. `volta_ptx_gemm.cu/.cuh`
contains the explicit `mma.sync.aligned.m8n8k4` candidate. Its fixed tiles
cover the large dense, small-M, and ragged cases, and its one- and two-stage
forms use synchronous global-to-register prefetch followed by shared-memory
stores. They do not use hardware asynchronous copies, TMA, or mbarriers.
`gemm_e09_smoke.cu/.cuh` reuses the Atlas E09 lane/slot coordinate map for a
short exact 256-coordinate check; it does not infer WMMA ownership.

The fused bias/ReLU path writes each output coordinate from the same lane and
slot that owns the GEMM accumulator. This identity ownership map does not
establish a shuffle-removal or second-MMA layout benefit. Its comparison is
the full fused operator against materialized FP32 C followed by a separate
bias/ReLU kernel.

`gemm_runner.cu` supports correctness, unprofiled CUDA-event timing, and
single-launch profiling modes. Correctness requires `--reference`; timing
includes the full selected device operator but excludes allocation and host
transfers. Native output includes the runtime cuBLAS API version queried from
its handle. The controller owns source/binary/input binding, foreground
admission, Compute Sanitizer, and Nsight Compute. Timing claims use only
unprofiled event samples. No source path changes device clocks or power
limits.

The pinned CUTLASS identity, license, and compatibility distinction are in
[`CUTLASS-PIN.md`](CUTLASS-PIN.md). The implementation is authored/adapted
code under the repository MIT license; CUTLASS is an include-only dependency
under its upstream BSD-3-Clause license and is not copied into this bundle.
