# KWV-08 publication outline

Target pages are `wiki/kernels/volta-gated-delta.md` and
`wiki/techniques/volta-state-ownership.md`, plus the measurement note
`sources/blogs/volta-recurrence-measurements.md` and the scoped artifact under
`artifacts/kernels/volta-recurrence/`.

## Kernel page

1. State the bounded operator exactly: `S` is `[d_v,d_k]`, `D=alpha*Sprev`,
   `e=v-D@k`, `S=D+beta*outer(e,k)`, and `o=S@q` reads the post-update state.
   List input/state dtype, gate and finite-value range, row-major layout, caller
   key-normalization responsibility, sequence lengths, padding and final-state
   contract.
2. Attribute the recurrence to the Gated Delta Networks algorithm source.
   Separate that algorithm source from locally compiled/executed SM70 evidence.
   Do not imply the source's released kernels were validated on V100.
3. Describe the per-token materialized baseline and one-kernel retained-state
   variant, the owner map, and the main resource risk (one `d_k` state row per
   lane may spill). Include only exact source-bound compiler/correctness results.
4. Link the measurement note and artifact; publish the shapes that passed and
   those that lost, with no generalized claim beyond measured shapes.
5. State that a chunked formulation was not tested unless the claimed KWV-08
   implementation supplies a cited derivation, actual caller availability of
   all chunk inputs, and complete setup/carry-inclusive measurements. Do not
   describe unavailable data or an unrun candidate as a performance rejection.

## State-ownership technique page

Explain why the same row's state feeds `D@k`, the rank-one update, and `S@q`;
then show the ownership map (sequence -> CTA, state row -> lane, `d_k` columns
-> lane-private state). Describe the retained-state tradeoff using measured
register/spill/occupancy evidence and complete timings. Keep source-level state
reads/writes separate from measured DRAM bytes. The page should not claim
Tensor-Core use or a chunk-parallel recurrence.

## Measurement note

Record the exact repository source and input hashes, compiler and flags, GPU
UUID/SM, controller execution IDs, sanitizer outcome, correctness errors for
all outputs and final state, and normal unprofiled complete-operator timing for
both variants. For the materialized baseline, timing includes T launches and
host enqueue cost; the comparison result covers that complete mapping. If a
profile is collected, bind its source/binary/report hashes and discuss only
counters that were actually present. A missing NCU counter is unavailable data,
not zero.

The note should distinguish semantic CPU oracle coverage from native GPU
correctness, native correctness from sanitizer evidence, and timing from the
profiled run. If retained state loses through spills/residency, keep the correct
per-token implementation and describe the measured boundary. If neither timing
nor the candidate proves an improvement, say that directly.
