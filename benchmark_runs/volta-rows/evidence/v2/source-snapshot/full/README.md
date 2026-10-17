# Volta row kernels

This artifact contains two standalone FP32 row operators for compute capability 7.0: stable softmax and plain RMSNorm. The kernels are intentionally local to these operators; this is not a general reduction library.

## Numerical and layout contract

Softmax computes `exp(x - row_max) / sum(exp(x - row_max))` over active columns. An optional byte mask contains only `0` or `1`; masked outputs are exactly zero, and an all-masked row returns all zeros. Logical finite inputs are bounded to `[-16, 16]`.

RMSNorm computes `gamma * x * rsqrt(epsilon + sum(x*x)/N)` with FP32 device accumulation and output. Its finite domain is `abs(x) <= 64`, `abs(gamma) <= 4`, and `epsilon` in `[1e-5, 1e-2]`. It has no mask, residual, bias, or fused producer. For `N <= 8193`, the sum-of-squares upper bound is `4096 * 8193`, below FP32 overflow.

Widths satisfy `1 <= N <= 8193`; a row count of zero is valid and skips the launch. Input, output, and optional mask have independent row strides and element offsets; strides are at least `N`, input padding is ignored, and output padding must remain untouched. The runner and fixture prohibit aliasing. Every logical output is compared to the independent Python binary64 reference computed from decoded represented FP32 inputs using `abs(actual-reference) <= 2^-30 + 2^-14*abs(reference)`. The input/reference bundle and contract contain exact hashes and per-case declarations.

## Device paths

`row_kernels.cu` implements a warp-per-row and a one-CTA-per-row softmax, plus warp, scalar CTA, and guarded `float4` CTA RMSNorm paths. All 32 lanes in a participating warp execute each synchronized shuffle. Tail elements and masks contribute reduction identities; CTA reductions use four shared partials and barriers. Zero-row cases do not launch.

The runner's fixed routes are:

| Operator | Baseline | Candidate |
| --- | --- | --- |
| Softmax | four warp-owned rows per 128-thread CTA | one 128-thread CTA per row |
| RMSNorm | scalar one-CTA-per-row | aligned `float4` one-CTA-per-row with scalar fallback |

The RMSNorm vector path is selected only after the actual post-offset input, gamma, and output addresses pass 16-byte alignment checks and both row strides preserve alignment. It processes complete groups of four and handles `N % 4` with a scalar tail. Otherwise, the dispatcher runs the scalar CTA. The native check and timing JSON report the actual selected path, including fallback and no-launch cases.

## Fixed workload and harness

`row_runner.cu` reads one combined 22-case fixture: correctness widths and edge cases for both operators plus a softmax single-row comparison, a larger softmax streaming case, and a matching RMSNorm streaming case. The selected streaming shapes have 512 rows by 4096 columns. Timing preallocates one or two alternating input/output sets, reports their byte counts and the current `cudaDeviceProp.l2CacheSize`, and labels only whether total working-set capacity exceeds L2. This capacity comparison does not establish actual cache residency or HBM traffic.

The same three timing cases are run for each implementation, preserving raw per-case samples. Setup copies use the same explicit nonblocking stream as the kernels and are drained before event timing; host-to-device setup is outside the measured interval. Correctness and sanitizer use the complete fixed bundle. The selected profile is one candidate softmax streaming launch; it is separate from unprofiled timing. The operator definitions exclude residual and bias, so this artifact makes no fusion claim. A future residual+bias result must compare the full fused operator against its full unfused producer/consumer path.

`benchmark_runs/volta-rows/source/prepare_rows.py` writes deterministic represented FP32 data, the independent row-major FP64 reference, a CPU-oracle record, and `KERNELWIKI-ROW-CONTRACT/1`. `row_oracle.py` validates the fixture and computes the independent reference. `test_row_oracle.py` contains CPU-only checks for deterministic encoding, masks, all-masked rows, tails, strides, offsets, zero rows, domain limits, truncation, output comparisons, and declared dispatch routes. The native runner reports aggregate and per-case checked, failed, non-finite, padding-write, and error/tolerance values with its GPU UUID and compute capability.

Build the native runner with the configured CUDA 12.9 toolkit using:

```sh
/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc \
  -O3 -std=c++17 -arch=sm_70 -lineinfo -Xptxas=-v \
  -Xcompiler=-fno-omit-frame-pointer \
  artifacts/kernels/volta-rows/full/row_runner.cu \
  -o benchmark_runs/volta-rows/bin/row_runner
```

Evidence runs use the Project Control CUDA controller launcher and role specifications under `benchmark_runs/volta-rows/controller/`; the launcher does not bypass controller admission. `PROVENANCE.yaml` binds the source and fixture files to the recorded build and validation evidence.
