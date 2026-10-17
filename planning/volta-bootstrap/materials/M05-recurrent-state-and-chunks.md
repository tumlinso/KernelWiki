# M05 — Recurrent state: preserve history while reducing movement

## Why this is included

Gated Delta Net and chunk parallelism already appear in KernelWiki. A Volta entry needs more than “replace the MMA.” It needs a correct state update, an appropriate low-query implementation, and an explanation of when chunked matrix work repays its overhead. This is one representative recurrence, not an attempt to build a general scientific dynamics toolkit. [KW-PRIMER, EXT-GDN]

## Explicit operator and reference

Use state `S` shaped `(d_v, d_k)`, key/query vectors of length `d_k`, and value vector length `d_v`. For each step:

```
D = alpha * S_previous
e = v - D @ k
S = D + beta * outer(e, k)
o = S @ q
```

Equivalently, `S = alpha*S_previous*(I - beta*k*k^T) + beta*v*k^T`. State orientation, whether output uses the pre- or post-update state, and where the gate is applied are part of the contract. The proposed operator uses the post-update state. Input keys may be normalized upstream; the harness must not silently normalize inputs if the caller supplied a different operator.

Keep an independent scalar/FP64 reference and a straightforward FP32-state native implementation. Initially use bounded finite inputs and an explicitly selected gate range. Preserve all output steps and final state. A candidate may not drop “old” tokens, reset the state at chunk boundaries, or share a state across independent sequences.

## Experiment S1 — State ownership and the two uses of D

The same state tile contributes to `D @ k` and is then updated. A materialized decomposition can reread or rewrite it several times. A fused row/tile owner can retain state through the reduction and rank-one update, subject to resource limits. Design the row/column reduction and output ownership together; accumulating only local pieces without the required cross-lane reduction changes the operator.

Compare a simple complete update with a fused state-tile variant and, where useful, a different tile shape. The potential saving is state movement and launch overhead; costs are registers, shuffle/shared reductions, and the longer live interval. No Tensor Core requirement is imposed on a low-batch update dominated by state traffic.

**Observe:** complete step time, state bytes read/written, reduction/instruction cost, registers/spills, and independent sequence/head parallelism. Test zero and nonzero initial state, a few state dimensions around native tiling boundaries, and enough independent states to expose both latency and throughput regimes.

**Falsifier:** keeping state live prevents useful residency or causes spills; a purported fused win omits final-state materialization that the caller requires. Retain the simple correct implementation where it is faster.

## Experiment S2 — A real chunk, not independent tiny products

For prefill with multiple known tokens, use an established algebraically equivalent chunk formulation from a pinned source or the paper. Derive its gate placement and state carry explicitly before adapting it to `Sm70`. Do not infer the recurrence from an attention kernel's similar shape. The atlas's four independent small MMA products do not magically implement a coupled recurrent system. [AT-R05, AT-E09, EXT-GDN]

Compare the sequential reference with one small and one larger practical chunk, not every chunk length. Include chunk preparation, local triangular/correction work, intermediate conversion and final-state carry. Chunking may increase matrix utilization but also add work and change floating-point accumulation order. Show both outputs and final-state error over the entire sequence.

For token-by-token autoregressive decoding, future inputs may not yet exist. Do not batch unavailable future tokens to make a benchmark fast. A chunk implementation is admitted only for callers that actually possess the chunk's inputs; decode and prefill are separate workloads.

**Falsifier:** setup/extra products dominate at short sequences, long-sequence error exceeds the contract, or the method relies on truncation or unprovided future inputs. Document that boundary rather than force a Tensor Core implementation.

## Tests that distinguish wrong state evolution

- `beta=0`: decay-only state evolution.
- `alpha=0`: no inherited state in that step.
- Repeated and orthogonal keys; mixed-sign values; nonzero initial state.
- The same sequence evaluated as one segment and as multiple segments with returned state carried forward.
- Long sequences with gates near one, plus varied gates and altered sequence order.
- Different independent sequences interleaved in memory, with ragged lengths and padding that must not update live state.

The tests are semantic, not merely “random output is close.” A chunk may differ numerically from sequential evaluation; tolerance must be justified, not relaxed after observing its errors.

## A useful reasoning bound—not a hardware claim

When `||k_t||_2 = 1`, `0 <= beta_t <= 1`, and `0 <= alpha_t <= 1`, the exact-real linear map on the inherited state has spectral norm at most `alpha_t`. Under a **separately justified** per-step numerical perturbation bound `epsilon_t`, state error satisfies

`||E_t|| <= alpha_t ||E_(t-1)|| + epsilon_t`.

This follows because `I - beta*k*k^T` has eigenvalue `1-beta` along k and one on orthogonal directions. The recurrence helps select stress cases and explain accumulated error. It does not establish `epsilon_t` for a proprietary MMA, cover unnormalized keys/out-of-range gates, or authorize history truncation. Near-one gates are especially important because they need not suppress earlier numerical error.

Use this reasoning to make tests discriminating; do not open a theorem-proving subproject or pretend finite measurements prove a global error bound. [AT-R15, EXT-FTTN]

## Publication outcome

One correct, benchmarked Volta recurrence and a state-ownership/chunking case study. Retain the reference even when a faster variant is found. If a chunk candidate is rejected, record the measured reason and leave a correctly scoped recurrence page—not a claim of a complete optimized GDN training implementation.
