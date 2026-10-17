# Volta / V100 routing

Use this route for V100 `sm70`. Select an exact target for executable guidance; a family-level source or a related modern page is not a compatibility guarantee. Use the existing query/fetch commands.

```bash
python3 scripts/query.py --architecture V100 --type hardware --compact
python3 scripts/query.py --architecture V100 --type kernel --compact
python3 scripts/get_page.py hw-volta-mma --follow-sources
python3 scripts/get_page.py kernel-volta-attention --follow-sources --include-code
python3 scripts/query.py --architecture volta --compact
```

## Current material

| Decision | Page |
|---|---|
| MMA/WMMA and fragment ownership | `hw-volta-mma` |
| Register/shared/warp tradeoffs | `hw-volta-resources` |
| Synchronization and progress | `hw-volta-participation` |
| Is a modern mechanism native on V100? | `hw-volta-feature-boundaries` |
| Accumulation or precision changes | `hw-volta-numerics` |
| A remaining operand/dependency bottleneck | `hw-volta-operand-delivery` |

## Native V100 kernel case studies

These exact-`sm70` entries link local source and the associated immutable
contracts and receipts. They describe measured implementations, not
universally optimal mappings:

| Family | Page | Result boundary |
|---|---|---|
| GEMM | `kernel-volta-gemm` | cuBLAS remains preferred for tested dense/small-M; ragged PTX comparison inconclusive, pinned CUTLASS won the tested ragged case |
| Row softmax/RMSNorm | `kernel-volta-row-kernels` | CTA softmax and aligned `float4` RMSNorm measured faster for three selected shapes |
| Attention | `kernel-volta-attention` | Prefill inconclusive; online decode slower; split16 faster only for the measured one-row D=70 case including merge; selected blocks correctness-only |
| Grouped/gated | `kernel-volta-grouped-gated` | Paired SIMT won balanced but lost skewed weighted grouping |
| W4A16 | `kernel-volta-w4a16` | Materialize then use dense cuBLAS for repeated resident cases; direct software decode is not a throughput winner here |
| Recurrent state | `kernel-volta-gated-delta` | Retained-state route won measured prefill; decode inconclusive; chunk implementation deferred |

The ordered GEMM host frontend puts uploads and computation on the same
explicit stream. Its [fresh correctness results](../benchmark_runs/volta-gemm-ordered/RESULTS.md)
cover six routes on each of the three GEMM shapes and both one-hot stage
counts. Use the [ordered frontend](../artifacts/kernels/volta-gemm-ordered/variants/README.md)
for new builds. Its ELF has no timing or profiler result; historical timing
claims in `kernel-volta-gemm` retain their original source and binary identity.

`scripts/get_page.py <page-id> --follow-sources` resolves cited source pages;
`--include-code` prints files under a page's `artifact_dir`. The SM70
hardware-concept pages intentionally have no native artifact directory. The
attention, GEMM, rows, grouped/low-bit, and recurrence pages expose their local
source bundles. For source-reported Hopper/Blackwell entries, inspect whether
the page has a vendored artifact directory before assuming complete code is
available; citations or snippets are not a runnable port.

Use native CUDA/CUTLASS/WMMA routes that actually compile and execute on the selected toolkit. A version number, a source header or a framework import is not a dispatch test. Keep the existing CUDA skill's host controller and numerical/measurement contracts. The user-supplied Volta NCU skill provides profiling; do not duplicate it here.

Preserve `sources`, confidence, reproducibility and precise performance
locators. Query results are navigation; explain the evidence and applicability
before recommending an implementation. Keep historical timings distinct from
new correctness-only receipts. Retain a correct library baseline when custom
code does not pay, and do not claim global optimality from this bounded study.
The compact family results and retrieval receipts are summarized in
`planning/volta-bootstrap/implementation/IMPLEMENTATION-HANDOFF.md`.
