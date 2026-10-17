# KWV-07 implementation boundary and numerical policy

## Grouped/gated

For every assignment `(row,expert,pos)`, the output is
`Y[pos] = silu(G) * U`, with `U = X[row] @ W_up[expert]` and
`G = X[row] @ W_gate[expert]`. X and both weight banks are represented FP16;
projections, intermediates, SiLU, product, and output are FP32. The candidate
must preserve duplicate assignments and scatter positions. Stable expert
ordering and offsets preserve empty groups.

The split baseline measures a stable host grouping pass, row gather, two
FP16-input/FP32-accumulating cuBLAS GEMMs per nonempty expert, FP32 up/gate
intermediate traffic, one SiLU/product epilogue, and output scatter. The paired
candidate is a deliberately bounded direct SIMT kernel for smaller M. It
reuses one assigned X row within each CTA/output-column tile and holds paired
FP32 projection accumulators. The comparison does not attempt a second
general GEMM engine and makes no Tensor Core fusion claim.

For each FP64 reference output, let `eU=gamma_K*sum_abs_up` and
`eG=gamma_K*sum_abs_gate`, where `gamma_K=(K*2^-24)/(1-K*2^-24)`. The
composite admission bound is

```text
|silu(G64)|*eU + (|U64|+eU)*(1+1/e)*eG
    + 8*2^-24*(1+|Y64|+|U64|+|G64|)
```

The final term is the fixed FP32 `expf`, division, and product allowance for
this bounded fixture suite. Every output is checked. It is a stated
engineering admission policy for these inputs, not a universal mathematical
error theorem.

## W4A16

The teaching storage format is signed two's-complement W4 with `q in [-8,7]`,
row-major `[K,N]` logical nibble order, even element in the low nibble, and odd
element in the high nibble. Scales are positive finite FP32 values indexed by
K groups and output column. Each decoded value is
`round_fp16_rn(round_fp32(scale*q))`; activations are represented FP16,
products accumulate in FP32, and outputs remain FP32. The tested scale range
is bounded by `[2^-14,8000]` and is chosen so every allowed code decodes to a
finite FP16 value.

The baseline materializes decoded FP16 weights then calls cuBLAS; the direct
candidate reconstructs each represented FP16 weight beside its SIMT dot
product. The first decoder pass and repeated compute use are reported
separately in device-event and synchronized host timing. Reuse batches invoke
the implementation 1/4/16 times at low M and 1/4 times at moderate M while
the same weights, scales, and activation remain resident and unchanged.
Cold full-path totals add input read/allocation/transfer setup and first decode
where applicable. The resident repetition is a declared benchmark condition;
capacity arithmetic does not demonstrate cache residency.

CPU tests cover all sixteen codes, odd packing tails, K-group tails, scale
rounding order, bounded scale extremes, negative and positive activations,
long-K cancellation, and fresh decode after weight/scale changes. The case
bundle has a distinct FP64 oracle over exact represented values. No native
INT4/NVFP4 arithmetic or complete checkpoint format is in scope.
