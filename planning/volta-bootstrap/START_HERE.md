# KernelWiki / Volta bootstrap

**Purpose:** make V100 (`sm70`) a first-class, performance-useful target of the existing KernelWiki, while preserving its upstream corpus and tools.

**Prepared:** 17 October 2026. **Delivery:** research, design, material seeds, and an implementation plan. **Not performed:** repository changes, Todo plan application, CUDA compilation, GPU execution, or performance qualification.

## Start here, then implement

1. Read [the execution handoff](IMPLEMENTER_HANDOFF.md) and [the design decision](design/01-extension-design.md).
2. Inspect [the native Todo plan](planning/kernelwiki.todo-plan.json). It contains a bounded set of implementation outcomes, not a new project-management framework. Validate and diff it with the installed Todo authority before applying it; follow [the intake procedure](planning/INTAKE.md).
3. Use [the coverage map](design/02-coverage-and-priorities.md) to find the relevant material brief. Begin with the supplied evidence and drafts instead of repeating the investigation.
4. Adopt `seed/` selectively, after architecture registration and native validation. These are source-backed draft additions, **not already-installed or hardware-qualified pages**.

The user will supply a working Volta port of **`~/ncu-report-skill`** before executing this plan. Use its actual installed interface for selected kernel profiles. Do not add an NCU port, a replacement profiler, or an autonomous optimization engine to this plan.

## What success means

An agent targeting V100 can retrieve applicable architecture facts, choose a sound CUDA/CUTLASS/WMMA route, find a correct representative kernel, understand why it is fast or limited, and follow a measured experiment to a better implementation. Existing Hopper/Blackwell queries and evidence remain usable. A negative optimization result is valuable when it explains a real boundary; it is not a reason to fabricate a win.

Coverage parity means comparable engineering usefulness—not identical instructions, page counts, benchmark scores, or a port of every named model. Modern hardware-specific features must be identified as unavailable, or replaced by an explicitly different software design where that is useful.

## Boundaries that matter

- Keep native `sources/`, `wiki/`, `queries/`, schemas and command entrypoints. Prefer small edits over a plugin loader, separate wiki engine, or architecture-wide schema migration.
- Retain the CUDA skill and Circuit-Bending Atlas as independent sources. Do not import the complete atlas or modify either repository as part of this work.
- Access those sources through Project Control **exact-path `read` calls**. Do not use semantic/scout queries against them.
- Keep index tests small and CPU-only. Spend GPU time on meaningful kernel decisions, not on elaborate infrastructure qualification.
- All shared-host GPU work uses the existing CUDA controller. No clock/power changes, privileged probes, edited cubins, or persistent background campaigns are authorized by this package.
- Establish numerical correctness and an honest baseline first. Profile the actual limiter. Explore a few discriminating alternatives, then stop or narrow scope when the result is clear.
- Do not silently turn an FP32-accumulating operation into FP16 accumulation, change quantization semantics, truncate recurrent history, or specialize to benchmark values.

## Package map

| File or directory | Use |
|---|---|
| `design/01-extension-design.md` | Chosen minimal patch, alternatives rejected, upstream update procedure |
| `design/02-coverage-and-priorities.md` | Existing KernelWiki topics mapped to honest Volta coverage |
| `design/03-validation-and-evidence.md` | Small infrastructure tests; kernel correctness and performance evidence |
| `materials/` | Concrete kernel/material designs, equations, baselines, tests and stopping conditions |
| `sources/` | External references, local exact-path inputs, source limitations and import decisions |
| `seed/` | Preassembled source pages, hardware pages and Volta routing reference |
| `planning/kernelwiki.todo-plan.json` | Native schema-v3 plan for the implementing agent |
| `planning/INTAKE.md` | Safe local validation and application procedure |
| `scripts/check_package.py` | Offline consistency check for this package only |
| `scripts/check_design_equations.py` | Optional CPU reference checks of the proposed equations; not GPU qualification |
| `validation/` | Actual package/equation check results and explicit unrun checks |

## Readiness and limits

The research/design can be used immediately. Runtime qualification remains implementation work. Earlier Project Control inspection reported `todo_read_database_open_failed` for `kernelwiki`; that is not evidence of an empty/uninitialized project. Do not create a replacement database or reset existing state to make this package apply. Resolve the existing authority's availability through its normal owner-maintained workflow.

The package checker is not the native Todo validator and does not compile kernels. Claims about those checks remain pending until they run on the implementation host.
