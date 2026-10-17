# M01 — Make data ownership feed Volta's matrix hardware

## Engineering thesis

The valuable port is not `tcgen05` renamed to another opcode. It is a decomposition in which the native multiplication produces data in a layout that the next operation can consume cheaply. A fast isolated MMA surrounded by conversion, shared-memory exchange and spills can lose to a less exotic implementation. This packet connects native matrix geometry, a useful GEMM baseline and a two-stage consumer. [AT-R05, AT-M24, AT-M25, SK-TENSOR]

## Established substrate to carry over

Explicit `mma.sync.aligned.m8n8k4` and C++ WMMA are different interfaces. The documented half-input explicit form computes four independent 8×8×4 products as a **full-warp collective**. The four logical groups are not independently launchable eight-thread warps. WMMA raw storage remains opaque; do not infer its ownership from the PTX map. [AT-R05, EXT-PTX, EXT-CUTLASS-MMA]

For the exact row-major-A, column-major-B, FP32-accumulator form already examined by atlas E09, the native coordinate map is a reusable starting point. E09 checked 256 distinct explicit-PTX coordinates. It did not establish raw WMMA lane ownership or every accumulator/layout variant. Reuse the experiment source and a short one-hot smoke test on the new binary; do not reopen the whole mapping investigation. A new FP16-accumulator path needs its own exact mapping check. [AT-E09]

CUTLASS's Volta multiplicand layout is an executable precedent for feeding the instruction. It is not interchangeable with a Hopper/Blackwell shared-memory swizzle. Inspect the pinned layout and the consuming iterator together. [EXT-CUTLASS-LAYOUT]

## Baseline and scope

Build a correct FP16-input, FP32-accumulating GEMM with a stated output conversion. Retain a cuBLAS baseline and one compatible CUTLASS `Sm70` baseline. A simple WMMA path is useful for teaching or as a control, not automatically a performance target. Own custom code only where layout, epilogue, small shape, or repeated glue gives it a reason to exist. [SK-TENSOR, SK-LOW-TENSOR]

Use a few shape regimes, not a Cartesian sweep: a large dense case, a skinny/small-M case, and a non-tile-multiple tail. Example starting points are M/N/K = 1024/1024/1024, 16/4096/4096 and 127/513/769, reduced when the host memory budget requires it. These are **proposed tests**, not historical measurements. Use nontrivial alpha/beta or bias where the selected operator includes them; otherwise do not silently change the equation.

## Experiment G1 — Buy reuse without starving the next tile

**Question:** does another software pipeline stage hide latency, or merely expand live registers/shared memory?

Compare the best simple tiled implementation with one additional prefetch/staging arrangement. Load next operands through supported register/shared-memory paths while doing independent current-tile arithmetic. Start with one versus two stages; try a third only if the profile still shows an operand-feed gap. Hold math, layouts and useful work fixed. A producer/consumer role split is an alternative only if it has a concrete advantage over unified warps.

Record normal kernel and complete-operator times; registers, spills, shared-memory use, eligible/active warps, memory traffic and the available stall indicators. Do not claim hardware `cp.async`, TMA or `mbarrier` support. Software overlap can consume the same registers and issue resources needed by the computation. [AT-R02, AT-R14]

**Falsifier:** the additional stage reduces stalls but worsens the full operator, introduces spills, or merely moves the bottleneck without a durable gain. Keep the simpler winner and document the limiting resource. No need to exhaust all tile sizes after the explanation is stable.

## Experiment G2 — The epilogue is part of the tile

**Question:** can the producer's accumulator ownership serve bias/scale/activation or a second multiplication without a canonical shared-memory round trip?

Compare three unlike candidates where meaningful:

- A canonical materialize/reload boundary.
- A register-and-shuffle ownership transform with the same arithmetic policy.
- A fused producer/consumer that retains the useful layout throughout.

For coordinate x, express producer ownership as `P(x)=(lane,slot)` and consumer ownership as `Q(x)=(lane,slot)`. The required movement is the explicit mapping from P to Q—not a matrix reshape. Count local slot changes, cross-lane words, shared stores/loads, barriers and conversions. Compose two successive mappings before emitting movement; permutations can cancel, but the cost of the resulting live state must be measured. This is a design method, not a promise that every pair of MMA layouts is compatible.

The main comparison uses the same FP32 accumulator contract. Treat FP16 accumulators as a separately named numerical variant. A variant that removes shuffles by changing arithmetic is not proof that the mapping alone is better. [AT-M25, EXT-SPARK, EXT-FASTATTN]

**Correctness:** one-hot inputs plus uniquely labeled accumulators; both producer and consumer outputs; ragged tails, nonzero bias, alternative input values and multiple launches. Every required lane executes its collective. Check all outputs, not only a checksum or finite values.

**Falsifier:** saved movement is outweighed by longer dependency chains, conversions or residency loss. Record the crossover by shape, rather than declare “register-only always wins.” This experiment supplies the ownership primitive used by attention and gated projections.

## Experiment G3 — Fix unused work before chasing peak FLOPs

Let useful work be `2MNK`. If the executed tiles pad dimensions, separately count padded products. Report useful throughput and the useful/executed ratio; never make a padded kernel look faster by crediting its unnecessary arithmetic.

For small-M work compare a shape-appropriate tile, batching independent matrices, and a SIMT/vector alternative. Split-K is considered only when output-tile parallelism is insufficient and the extra partial-output/reduction traffic can be repaid. Match the numerical order/tolerance explicitly. Do not use split-K just because it exists in a modern library.

A useful approximate cost test for a transformation reused r times is:

`T_new = T_prepare + r * T_new_step + T_finalize`.

It has a chance to win only when `r * (T_baseline_step - T_new_step) > T_prepare + T_finalize`. If the denominator is non-positive, no reuse count rescues that measured transformation. Use this to decide which cases merit a run; do not publish the model as a measured crossover.

## Conditional deeper work

Only if the retained kernel is operand-/issue-limited after layout and memory fixes, inspect the named hot SASS region and try a source-level accumulator ordering or unroll change. Jia's historical register-bank observations motivate a hypothesis; they are not a current compiler guarantee. Do not port a cubin editor or conduct all-instruction latency sweeps. See M06.

## Publication outcome

A native-matrix hardware page; a fragment/epilogue technique page with real compilable code; one GEMM case study with the best retained baseline, measured boundaries, and source/binary identities. Publish failed stage/layout alternatives when they teach a concrete constraint. No universal performance target or “fully optimized” label is required.
