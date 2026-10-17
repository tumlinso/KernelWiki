# Volta / V100 routing

Use this route for V100 `sm70`. Select an exact target for executable guidance; a family-level source or a related modern page is not a compatibility guarantee. Use the existing query/fetch commands.

```bash
python3 scripts/query.py --architecture V100 --type hardware --compact
python3 scripts/get_page.py hw-volta-mma --follow-sources
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

Add kernel and technique IDs here only after their pages/artifacts exist. The material briefs in the bootstrap define the intended GEMM, row, attention, grouped/low-bit and recurrence investigations; they are not yet runnable wiki entries.

Use native CUDA/CUTLASS/WMMA routes that actually compile and execute on the selected toolkit. A version number, a source header or a framework import is not a dispatch test. Keep the existing CUDA skill's host controller and numerical/measurement contracts. The user-supplied Volta NCU skill provides profiling; do not duplicate it here.

Preserve `sources`, confidence, reproducibility and precise performance locators. Query results are navigation; explain the evidence and applicability before recommending an implementation. Retain a correct library baseline when custom code does not pay.
