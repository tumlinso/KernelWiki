# M06 — Spend deeper profiling only where it changes the kernel

These are conditional investigations inside the kernel tasks, not additional mandatory projects. Start them only when the retained implementation's measurements leave the named question open. Reuse the ported NCU skill, focused compiler output and the atlas's established constraints.

| Trigger | Discriminating change | Keep controlled | Decision / stopping rule |
|---|---|---|---|
| Good locality but few eligible warps and a dependency-heavy hot loop | Reorder independent accumulators; introduce enough independent required work to test ILP | Same useful operations, precision, global traffic and input set | Retain only if normal operator time improves; avoid claiming a bank cause from timing alone |
| Suspected operand-delivery problem in one hot instruction region | Inspect SASS operands; compare a source-level scheduling/layout change with matched register budget | Instruction semantics and dependency graph as far as practical | Jia's historical parity-bank result is a hypothesis source, not a guaranteed mapping on this compiler |
| Aggressive unrolling grows code with no gain | Compare modest unroll choices and code size/instruction-fetch indicators | Work, layout and major occupancy regime | Stop at the small Pareto set; do not sweep every instruction or build a new compiler |
| Fused softmax/activation dominates after memory fixes | Try an explicitly bounded approximate-exp or refinement alternative | Declared domain, maximum/per-output error and downstream operator | Reject if accuracy contract changes silently or full operator gain is negligible |
| Too much shared traffic between two compatible stages | Compose their ownership maps, then compare register/shuffle and shared routes | Exact producer/consumer mapping, not only final shape | Carry the winner into the actual fused kernel; a standalone shuffle micro-win is insufficient |
| Grouped workload has a measurable heavy tail | Compare bucketing versus finite persistent work | Same groups, duplicate semantics, admission and termination | Stop if scheduler cost exceeds recovered tail time; no general resident runtime |
| Packed storage reduces bytes but runs slowly | Separate decoding, address/request and arithmetic costs | Same decoded values and scale order | Choose decode location by batch/reuse; retain negative regimes |

## Why not repeat the atlas campaign

The atlas already answers some substrate questions and carefully limits others. E09's explicit FP32 MMA mapping is reusable. E06's bank-pattern timing does not establish bank causality; E02 does not isolate instruction latency; E04 and restricted E27–E30 remain unrun. E11's bitset result does not imply tensor counting is generally faster, and E39 has no established arithmetic crossover. Use these limits to avoid false premises—not as a checklist requiring every missing experiment to be completed. [AT-E09, AT-E06, AT-LIMITS]

An experiment is warranted when at least one of its possible results changes a current implementation choice or a material claim. Record that choice before running it. When available counters cannot distinguish mechanisms, publish the measured performance boundary and label the causal explanation uncertain. Do not spend a week building instrumentation for a claim that the wiki does not need.

## A practical budget rule

Start with the strongest conventional baseline and two materially different candidates. Profile one representative bottleneck case. Expand only the dimension that explains a real crossover or correctness failure. Keep successful and rejected results under the same numerical contract. This is a starting discipline, not a rigid cap when an important result genuinely needs another test.

Never equate maximum occupancy, peak power, nominal Tensor Core utilization, or a low-level opcode with a useful end-to-end improvement. The product being built is actionable knowledge about **which mapping wins under which conditions**.
