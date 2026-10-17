---
id: blog-volta-grouped-lowbit-measurements
title: 'Volta grouped gated projections and W4A16 decoding'
author: KernelWiki Volta grouped/low-bit investigation
url: https://github.com/tumlinso/KernelWiki/blob/master/benchmark_runs/volta-grouped-lowbit/summary.json
source_category: benchmark-blog
architectures: [sm70]
tags: [grouped-gemm, gated-dual-gemm, fine-grained-quantization]
retrieved_at: '2026-10-17'
---

# Scope

This is a first-party native investigation on one Tesla V100-SXM2-16GB
(`GPU-21131915-1488-23af-38dd-1743ae1f5cc8`). It covers two deliberately
bounded operator questions: whether a paired small-M SIMT kernel can beat
stable grouping plus two cuBLAS projections and a gated epilogue, and whether
W4A16 decoding should be materialized before a dense GEMM or performed near a
small-M dot-product consumer.

The immutable represented inputs, FP64 references, numerical contracts, CPU
identity/precheck records, controller specifications, exact source closure,
and controller output captures are under
[`benchmark_runs/volta-grouped-lowbit/`](../../benchmark_runs/volta-grouped-lowbit/README.md).

# Fixed numerical contracts

The grouped operation evaluates `U=X[row]W_up[expert]`,
`G=X[row]W_gate[expert]`, then writes `silu(G)*U` at the assignment's explicit
output position. Inputs and weights are represented FP16; each projection,
the SiLU/product epilogue, and output are FP32. Duplicate `(row,expert)`
assignments remain separate. Empty experts and skew are represented in the
input and stable host grouping map.

The W4A16 storage contract uses signed two's-complement codes in `[-8,7]`,
row-major `[K,N]` nibble packing (even code in the low nibble), and a positive
finite FP32 scale per 128-element K group and output column. Decode is
`round_fp16_rn(round_fp32(scale*q))`; activations and decoded weights are
represented FP16, dot products accumulate in FP32, and outputs are FP32. The
binary FP64 references are generated from the exact represented inputs before
device work. The suite-level CPU oracle records bind those exact reference
files and only attest to CPU precheck results; native output status is
reported by each captured runner role.

# Comparison protocol

The grouped suite has six assignment distributions across small, moderate,
and weighted matrix regimes. The split route includes stable grouping,
gather, both projections for each nonempty expert, FP32 intermediates, the
SiLU/product epilogue, and output scatter. The paired route is a small-M SIMT
candidate. Resident event samples exclude one-time allocation and H2D setup;
separate host values include assignment preparation and one-time harness input
setup.

The W4 suite retains an odd K/N tail fixture and adds the same 2 MiB packed
weight matrix for `M=1` and `M=16`, `K=4096`, `N=1024`. At that size the packed
codes and FP32 scales total 2.125 MiB, while expanded FP16 weights take 8 MiB.
This is representation capacity arithmetic only, not proof of HBM or cache
traffic. Actual repeated batches compare reuse counts 1/4/16 at low M and 1/4
at moderate M while the weights, scales, and activations stay resident and
unchanged. Predecode and repeated-compute medians are separate measurements;
derived prep-plus-batch device and setup-plus-prep-plus-batch host estimates
are arithmetic on those measured components.

# Native results

The source-bound v1 ELF (`sha256=c9eae1a8c7ceac582d5a39e8595656068797b15b57559ae5ddd3b8ff2dd10d88`) was built once for `sm_70` with CUDA 12.9 and reused by all twelve Project Control roles. The frozen source-closure manifest is SHA-256 `196094fbf457b3da40b9c790526ba072465c2ac6989e2ea66c0d4a1f815da14f`. The GPU was a Tesla V100-SXM2-16GB, compute capability 7.0, UUID `GPU-21131915-1488-23af-38dd-1743ae1f5cc8`.

Both implementation checks passed every output against the fixed admission bound: 70,144 grouped outputs and 18,479 W4A16 outputs per role, with zero failed or nonfinite results. Each suite also ran candidate Compute Sanitizer memcheck. The raw logs report `ERROR SUMMARY: 0 errors`, but the Project Control wrapper classified both runs as partial and inconclusive, so they are not reported as complete sanitizer qualification. The two focused NCU captures have one selected-kernel sample each and valid counters, but `timing_valid=false`; they are diagnostic only and are excluded from all timing results.

## Grouped/gated results

The table reports the median CUDA-event time for the device event scope and the median synchronized host scope. Each median uses nine positive samples. `setup` covers input and FP64-oracle reads/parsing, validation, host assignment grouping, allocations, H2D, and synchronization. The group-metadata timer is a subcomponent of setup, not an extra cost. `setup + warm host` is a derived one-use sum, not a separately measured application request.

| Case | Event median, split / paired (µs) | Synchronized host median, split / paired (µs) | Setup, split / paired (µs) | Derived setup + warm host, split / paired (µs) |
|---|---:|---:|---:|---:|
| tiny balanced | 60.416 / 13.312 | 68.524 / 20.006 | 455.636 / 445.512 | 524.160 / 465.518 |
| tiny skewed | 35.840 / 13.312 | 42.826 / 19.890 | 384.000 / 372.521 | 426.826 / 392.411 |
| moderate balanced | 63.488 / 20.480 | 70.144 / 27.098 | 1,667.105 / 1,701.955 | 1,737.249 / 1,729.053 |
| moderate skewed | 48.128 / 20.480 | 54.544 / 27.369 | 1,566.026 / 1,579.985 | 1,620.570 / 1,607.354 |
| weighted balanced | 190.464 / 69.632 | 198.305 / 76.667 | 40,708.444 / 41,237.950 | 40,906.749 / 41,314.617 |
| weighted skewed | 45.056 / 66.560 | 52.749 / 73.920 | 39,434.429 / 40,332.861 | 39,487.178 / 40,406.781 |

Paired SIMT won the measured tiny, moderate, and weighted-balanced cases. The weighted-skewed case reversed the ranking: it has only two nonempty experts, compared with eight in the balanced case, which changes the split route's cuBLAS call count and per-call row sizes. The paired event time stayed nearly flat between weighted distributions (69.632 and 66.560 µs), while split cuBLAS moved from 190.464 to 45.056 µs. This supports dispatch based on actual group sizes and count; it does not isolate fusion as the sole cause because the implementations use different GEMM engines. Host group-metadata preparation was 0.672–4.837 µs inside setup and outside the CUDA event. The very large weighted setup figures come from this harness's input and oracle I/O, and should not be read as production startup latency.

The CUDA event covers gather, both up and gate projections for each nonempty expert, the FP32 SiLU/product, and scatter. Stable assignment grouping and metadata preparation are host setup, outside the event. Synchronized host timing covers submission through completion. The derived host column is `one_time_setup_host_us + median_host_us`; it is arithmetic on separately recorded setup and warm-use timings, not an end-to-end startup-latency measurement.

## W4A16 decode and reuse results

The materialized implementation measures predecode separately and reports the median event and host durations, then separately measures repeated dense cuBLAS batches. The direct implementation has no decode step and measures repeated fused-decode SIMT batches. In each row, “prep + batch event” is the sum of the measured predecode median and the measured repeated-batch median (or just the direct batch median); it is not a single event interval. Reuse counts are ordered 1/4/16 for `M=1` and 1/4 for `M=16`.

| Case (`M,K,N`) | Predecode median event / host (µs) | Dense batch median event / host (µs) | Direct batch median event / host (µs) | Derived prep + batch event, materialized (µs) | Direct batch event by reuse count (µs) |
|---|---:|---:|---:|---:|---:|
| tail `1,257,63` | 6.144 / 12.620 | 10.240 / 17.541 | 53.248 / 60.156 | 16.384 / 36.864 / 116.736 | 53.248 / 203.776 / 805.888 |
| tail `16,257,63` | 5.120 / 12.596 | 14.336 / 20.904 | 54.272 / 61.057 | 19.456 / 39.936 | 54.272 / 207.872 |
| weighted `1,4096,1024` | 45.056 / 52.816 | 27.648 / 34.662 | 809.984 / 816.976 | 72.704 / 132.096 / 371.712 | 809.984 / 3,229.696 / 12,906.496 |
| weighted `16,4096,1024` | 45.056 / 52.838 | 31.744 / 38.821 | 772.096 / 779.641 | 76.800 / 146.432 | 772.096 / 3,080.192 |

The derived device values above correspond to reuse counts 1/4/16 or 1/4, as applicable. For a view of the host setup cost, the separately derived `setup + predecode host + batch host` model is:

| Case | One use, materialized / direct (µs) | Maximum measured reuse, materialized / direct (µs) |
|---|---:|---:|
| tail `1,257,63` (`r=16`) | 225.317 / 244.556 | 326.097 / 997.508 |
| tail `16,257,63` (`r=4`) | 369.229 / 361.056 | 390.317 / 514.451 |
| weighted `1,4096,1024` (`r=16`) | 16,523.163 / 16,980.054 | 16,822.347 / 29,076.641 |
| weighted `16,4096,1024` (`r=4`) | 17,850.182 / 18,909.234 | 17,920.000 / 21,217.343 |

These host models add harness setup, a measured predecode host time where applicable, and the median host duration of the measured batch. Setup includes fixture and FP64-oracle reads, validation, allocations, H2D, and synchronization under an initialized CUDA context. It is not a directly timed cold production call. Within each case, the activation, packed-code, and scale bytes are held unchanged across reuse; the `M=1` and `M=16` weighted fixtures share byte-identical packed codes/scales, while their activations are independent and are not prefix-matched.

For the weighted matrix, packed codes plus scales occupy 2.125 MiB and the expanded FP16 matrix occupies 8 MiB. That is representation capacity arithmetic only and establishes no L2 or HBM residency. Materialization plus dense cuBLAS had the lower resident compute time on every tested shape. The direct route remains a correct, smaller-storage alternative, but it did not provide a throughput win here. No native INT4/NVFP4 or checkpoint-compatibility claim follows from this software format.

All machine-readable timings, per-case correctness identities, exact role specs, captured controller exports, and profile artifacts are bound in [`benchmark_runs/volta-grouped-lowbit/summary.json`](../../benchmark_runs/volta-grouped-lowbit/summary.json). CPU precheck/oracle files remain distinct from the native PASS records in those role captures.
