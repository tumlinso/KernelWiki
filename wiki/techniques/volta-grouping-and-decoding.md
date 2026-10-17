---
id: technique-volta-grouping-and-decoding
title: Grouping Work and Decoding Packed Weights on Volta
type: technique
architectures: [sm70]
tags: [tile-scheduling, data-reuse, kernel-fusion, fine-grained-quantization]
confidence: experimental
reproducibility: benchmarked
prerequisites: [kernel-volta-gemm]
related: [kernel-volta-grouped-gated, kernel-volta-w4a16, technique-volta-memory-fusion]
sources: [blog-volta-grouped-lowbit-measurements]
performance_claims:
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 represented inputs/weights; FP32 projections and gated epilogue
    shape: 128 assignments, 8 experts, K=512, N=256, balanced groups; paired SIMT
    metric: CUDA-event median latency (us)
    value: 69.632
    source_id: blog-volta-grouped-lowbit-measurements
    source_locator: "Native results > Grouped/gated results > weighted balanced > paired event median"
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP16 activations; W4A16 decoded weights; FP32 dot output
    shape: M=1, K=4096, N=1024; direct signed-nibble decode
    metric: Median resident CUDA-event batch latency (us)
    value: 809.984
    source_id: blog-volta-grouped-lowbit-measurements
    source_locator: "Native results > W4A16 decode and reuse results > weighted 1,4096,1024 > direct batch median event"
blackwell_relevance: Grouped scheduling and decode-placement costs remain workload-specific; newer native low-bit arithmetic changes the representation and compute boundary and requires its own measurements.
---

# Group by useful work, not by assumption

When several small matrix products share the same shape, an explicit stable
assignment-to-expert map can recover simpler dense library calls without
losing identity. Keep every assignment, preserve duplicates, and scatter by an
explicit output position. Empty groups belong in the metadata so the mapping
can be checked. The full comparison includes host metadata work, input gather,
all projections, intermediate materialization, epilogues, and scatter.

For a gated pair, a fused CTA can stage one assigned input row once and hold
both projections' accumulators together. This saves a launch and intermediate
traffic, but adds live accumulator state and can lose the library's matrix
tile. Measure small and weighted groups, balanced and deliberately skewed,
under the same numerical contract. Test a finite persistent queue only after a
measured heavy tail remains after ordinary stable grouping; charge queue
construction, metadata transfer, termination, and output mapping to that
candidate.

The KWV-07 V100 measurements show why the grouping policy must account for
distribution: paired SIMT was faster in the weighted balanced fixture, but
the split cuBLAS route was faster in the weighted skewed fixture, where only
two experts were nonempty. The small and moderate fixtures favored the paired
route. These measurements compare complete operator routes with different
GEMM engines; they do not isolate the effect of the fused epilogue by itself.
No persistent queue was tested because this bounded comparison did not
establish a tail bottleneck that justified it.

# Put low-bit decode at the right boundary

For an exact storage format, compare measured `T_decode` and an `r`-use dense
batch against the corresponding `r`-use direct batch. Include weights/scales
and activation setup, input transfers, decode, and each dense or fused compute
call. In KWV-07, decode and reuse batches were measured separately; reported
prep-plus-batch values add the decode median once to the repeated-batch median
and are not a directly timed cold interval. Host setup includes file reads for
the represented input and FP64 oracle, validation, allocation, H2D, and sync
under an initialized CUDA context. The resulting setup-plus-warm-use host
model is not production startup latency. A decoded cache is a separate
lifetime/capacity contract. Storage byte counts alone do not show which memory
tier supplied the data.

For the measured weighted shape, the materialized route decoded in 45.056 µs
and then ran the dense `M=1` batch in 27.648 µs, while direct decode took
809.984 µs for its one-use batch. The `M=16` numbers were 45.056 + 31.744 µs
versus 772.096 µs. This supports predecode for repeated use on this software
path; it does not establish a native low-bit instruction path or a general
Volta dispatch rule. The direct route still stores fewer weight bytes.

The direct small-M dot path rounds each scale/code product to FP16, then
accumulates its product with the represented activation in FP32:

```cuda
const int q = signed_code(packed_codes, weight_index);
const float scale = scales[static_cast<std::size_t>(group) * n + col];
const __half w_half = __float2half_rn(scale * static_cast<float>(q));
sum = fmaf(x, __half2float(w_half), sum);
```

Rounding order is part of the format: multiply a represented FP32 scale by the
signed code in FP32, then round that product to FP16. Validate all code values,
odd nibble packing, the K group tail, scales near representational limits,
negative activations, cancellation, and changed weights/scales on a later
call. Compare outputs against an FP64 oracle over the exact represented
inputs. The [KWV-07 report](../../sources/blogs/volta-grouped-lowbit-measurements.md)
contains the measured shape/reuse boundary and its limits.

Verify the immutable represented-input bundle and its FP64 references without
regenerating them:

```sh
python3 benchmark_runs/volta-grouped-lowbit/source/bundle.py verify \
  benchmark_runs/volta-grouped-lowbit/cases/fixture_bundle
```
