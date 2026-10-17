# M02 — Reduce traffic and requests without losing residency

## Why this belongs before exotic instruction work

A representation can waste bandwidth, memory requests, address instructions, or registers independently. High nominal HBM bandwidth and low measured bandwidth do not identify the same problem. Trace useful bytes, actual traffic, requests and the next consumer. Atlas E06–E08 report useful observations but do not establish bank/TLB causes or a whole-pipeline winner. [AT-R03, AT-E06, AT-LIMITS]

## Representative operators

Use a row softmax plus one normalization operator. RMSNorm is a compact default:

`r = rsqrt(epsilon + sum_j x_j^2 / n)`; `y_j = gamma_j * x_j * r`.

State where reduction and multiplication occur in FP32 and where input/output conversion occurs. Include residual+bias before normalization only as a separately defined fused operator. If LayerNorm is already available, retain its stable mean/variance algorithm; do not replace it with an unstable variance shortcut to get speed.

Use rows near warp/tile boundaries and at a few useful widths: e.g. 31/32/33, 128, 768, 4096, plus one awkward wide width such as 8193. Small row counts expose insufficient parallelism; larger row counts expose throughput. Not every width needs every variant or a profile. Independent CPU FP64 checks cover small inputs; compatible CUB/library or already-correct native kernels supply performance baselines.

## Experiment R1 — Fewer instructions or fewer bytes?

Compare scalar/coalesced access to aligned vector access while holding the semantic addresses and output fixed. A wide load may reduce address and request pressure even when DRAM bytes are unchanged. It does not necessarily reduce the minimum data volume. Preserve a scalar/tail path where alignment or extent requires it. Do not violate aliasing or alignment contracts through reinterpret casts.

**Observe:** normal timing, generated load widths, request/transaction metrics where available, executed instruction count, useful and actual bytes, and eligibility/stall changes. A bandwidth number alone is not the explanation.

**Falsifier:** the wide path simply pads/overreads, increases registers enough to lose throughput, or only wins for an alignment the operator cannot guarantee. Publish dispatch requirements rather than overgeneralize. The current community SM70 KV work is a useful candidate example of request reduction, not an architecture-wide speed guarantee. [CAND-1CAT]

## Experiment R2 — Registers, shared memory, or a second pass?

Compare a warp-local row path, a CTA cooperative path, and a split/reload path only in width regimes that could favor them. The trade is saved reads versus live values, shared use and the number of independently eligible warps. A register cap is a diagnostic perturbation, not a goal. A shared carveout consumes another part of the local memory hierarchy; account for the whole producer/consumer. [SK-REG, SK-FUSION, AT-R03]

Use a controlled producer/consumer permutation to compare padding and the appropriate swizzle. Include both stages and their barriers in the timed boundary. Reuse E06's semantic controls, but collect supported conflict/traffic counters when the causal question is bank behavior. Equal-looking microsecond timings at the launch/timer floor are not a performance result—batch enough useful repetitions without making the compiler remove the work.

**Falsifier:** lower occupancy is not itself failure; failure is a worse full path or a explained stall/spill cost that erases the saved reads. Likewise, more occupancy is not success if it adds materialization passes.

## Experiment R3 — Fuse the actual operator boundary

For residual-plus-normalization or softmax-plus-output use, compare the same full sequence fused and unfused. Count each removed intermediate write/read, but also include the longer live ranges, register allocation and any extra reductions. Keep a graph-replayed unfused baseline only if launch overhead is material; graphs cannot remove intermediate HBM traffic.

A simple cost model is:

`saving ≈ removed intermediate bytes / achievable bandwidth + removed launch cost − added computation/synchronization/residency cost`.

This selects a plausible candidate, not a substitute for timing. Report resident operator time and a separate boundary including unavoidable conversion/preparation. Allocation/setup is included only when the intended caller pays it; name exclusions.

## Correctness details worth the test time

Capture warp membership before divergence and use valid shuffle sources. Partial rows must not use a full-warp reduction mask as a license to read inactive values. Validate normalization for zero rows, varied scales, outliers, cancellation-sensitive LayerNorm inputs where relevant, and alternate inputs after warmup. In-place operation is supported only when its actual aliasing contract is safe.

For softmax, use a stable maximum/normalizer and define all-masked behavior explicitly. Do not let `-infinity - -infinity` create NaNs accidentally. The initial contract may support finite inputs plus explicit masked positions while rejecting arbitrary NaN/+infinity inputs; say so rather than claim universal handling.

## Publication outcome

One row-kernel case study and one Volta memory/fusion diagnostic route with thresholds described by measured regimes, not global occupancy rules. A couple of retained variants can share one source harness. Do not build an exhaustive reduction library or transfer/NUMA benchmark suite.
