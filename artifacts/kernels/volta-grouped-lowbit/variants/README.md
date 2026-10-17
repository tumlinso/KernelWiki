# KWV-07 grouped/gated and W4A16 sources

This source bundle compares two bounded operators on native SM70. It is an
operator study, not an MoE runtime or a general quantization implementation.

`kwv07_runner.cu` provides the native correctness and timing harness. Its
split grouped/gated path stably groups assignment indices, gathers each
assigned row once, runs separate FP16-input/FP32-accumulating cuBLAS
projections for each nonempty expert, and applies the FP32 SiLU/product and
scatter. `grouped_lowbit.cu` adds a small-M paired SIMT candidate which loads
each input row once per output tile and computes gate/up projections together.
It is not a tensor-core kernel.

The W4A16 paths in `grouped_lowbit.cu` compare a signed-nibble decode to
represented FP16 followed by cuBLAS against direct decode beside each FP32
dot-product term. `group_size=128`; the contract is defined by the exact
contracts and independent CPU references in `benchmark_runs/volta-grouped-lowbit/`.

All native copies and kernels use one explicit nonblocking stream. Pageable
host uploads are queued with `cudaMemcpyAsync` on that stream and synchronized
before their source buffers can leave scope. D2H correctness copies use the
same stream and complete before the host reads them. The timing roles exclude
initial allocation/transfers from device-event samples and report one-time
setup separately from synchronized host compute time.

The native runner checks every selected output against a predeclared
per-output FP64-reference admission bound. It emits
`KERNELWIKI-KWV07-NATIVE/1` PASS metadata only after all checked outputs pass;
the CPU suite oracle JSON is an input identity and arithmetic precheck, not a
native result. The controller-captured build and role records are in
`benchmark_runs/volta-grouped-lowbit/evidence/v1/`.

Run only the host reader checks without CUDA:

```sh
make -C artifacts/kernels/volta-grouped-lowbit/variants -f Makefile.cpu cpu-test
```

The `Makefile.cuda` recipe is narrow to `sm_70` and CUDA 12.9. Controller role
specifications build the shared executable exactly once in the first grouped
correctness role and reuse those bytes for every later role.
