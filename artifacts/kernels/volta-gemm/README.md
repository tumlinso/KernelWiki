# Volta GEMM implementation bundle

This directory contains the source-bound FP16-input, FP32-accumulation GEMM
investigation for SM70. The native source and implementation details are in
[`variants/README.md`](variants/README.md); measured behavior and limitations
are summarized in the [kernel page](../../../wiki/kernels/volta-gemm.md) and
[measurement report](../../../sources/blogs/volta-gemm-measurements.md).

The serialized contracts, represented inputs, independent CPU references,
controller role specifications, and measured binary are under
[`benchmark_runs/volta-gemm`](../../../benchmark_runs/volta-gemm/README.md).
The recorded comparison uses row-major A[M,K] and B[K,N], FP16 inputs, FP32
accumulation/output, alpha=1, beta=0, and optional FP32 bias followed by ReLU.
The measured shapes are dense, small-M, and ragged; results apply only to
those contracts and the recorded V100/CUDA configuration.

For evidence-grade reruns, use the repository's CUDA controller and the
saved role specifications in `benchmark_runs/volta-gemm/controller/`. Do not
interpret a direct runner invocation as controller-admitted evidence. The
saved specifications preserve their original host paths; the CUTLASS source
identity and license are recorded in [`variants/CUTLASS-PIN.md`](variants/CUTLASS-PIN.md).
