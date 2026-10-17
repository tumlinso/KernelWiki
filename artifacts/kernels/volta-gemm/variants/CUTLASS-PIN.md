# CUTLASS Sm70 baseline pin for KWV-04

Use NVIDIA CUTLASS tag `v2.11.0` as the fixed Sm70 baseline source. Its own
support documentation covers CUDA 11.x; this repository's local CUDA 12.9
build result is a separate, bounded observation.

- Upstream: `https://github.com/NVIDIA/cutlass`
- Exact tag: `v2.11.0`
- Resolved commit: `66d9cddc832c1cdc2b30a8755274f7f74640cfe6`
- Checkout: `/tmp/kernelwiki-gemm-prep.X4Sd0K/cutlass-v2.11.0`
- Checkout state at inspection: clean detached shallow tag checkout
- License: BSD-3-Clause, from `LICENSE.txt`
- `LICENSE.txt` SHA-256: `5f0ea053230eb92267e9e80ed67ec08899ebc55d093eff4a2273a7b340a48cca`
- `include/cutlass/arch/mma_sm70.h` SHA-256: `f670a45ffc54b5b9487104d173b3695f822ebff3311b00f357ae1200fde79903`
- `include/cutlass/layout/tensor_op_multiplicand_sm70.h` SHA-256: `4ba44a947e82899c760913f79e5def50b86c610c0c20652fec4aac9ef73e8101`

The pinned source contains a concrete FP16/FP16, FP32-accumulator/output
`device::Gemm` instantiation with row-major A, B, and C, `arch::Sm70`,
instruction shape 8x8x4, CTA shape 64x64x32, warp shape 32x32x32, and two
stages in
`test/unit/gemm/device/gemm_f16t_f16t_f32t_volta_tensor_op_f32_sm70.cu`.
The checked source lines instantiate precisely those types. The task wrapper
follows that configuration, with AlignmentA/B=1 and epilogue vector width 1
so the odd K/N strides in the ragged case are admitted. It uses the explicit
nonblocking controller stream. A controller build under the local CUDA 12.9
compiler has succeeded, including this Sm70 template instantiation. This is
build evidence only; actual `can_implement`, runtime correctness, sanitizer,
and matched controller timing are still required before calling it a
qualified baseline.

**Compatibility distinction:** CUTLASS 2.11.0's own README describes CUDA
11.x compatibility and recommends CUDA 11.8. The observed 12.9 compile is a
bounded local result, not a broader vendor compatibility promise. Whether
the instantiated baseline is implementable and correct on this V100 remains
subject to the native controller runs. No lowering of the toolkit or
dependency substitution is implied.
