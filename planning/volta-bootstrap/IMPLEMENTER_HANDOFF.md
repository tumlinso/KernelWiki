# Implementation handoff

## Objective and authority

Execute the supplied native Todo plan in the **standalone `kernelwiki` workspace** using the current Project Control task lifecycle. This package is the research/design input, not a mandate to replan the whole system. Keep implementation, tests and commits there. KDA receives a tested dependency commit later.

The user explicitly prioritizes practical Volta kernel engineering over infrastructure ceremony. The design is settled enough to implement. Change it only for a concrete conflict with the current repository or evidence, and record the smallest justified adjustment. Do not reopen broad architectural research, introduce a plugin system, or replace native tools.

## First actions

Place the complete package at `planning/volta-bootstrap/` inside the KernelWiki checkout, preserving its internal relative paths. Validate/diff its native plan with the installed authority before applying it; see `planning/INTAKE.md`. Preserve existing uncommitted work and task claims. Record HEAD/dirty paths and check the small shared surfaces against the supplied source hashes.

Read `design/01-extension-design.md`, then the current `AGENTS.md`. Source research and architecture registration do not require GPU work. Use the source ledger and each material's explicit inputs to avoid redoing solved work.

The user guarantees a usable Volta port at **`~/ncu-report-skill`** before this implementation begins. Read its actual entrypoint and bind it to the existing CUDA controller. A failed environment check blocks only the dependent GPU work. Do not turn this bootstrap into the NCU port or reset Todo state to bypass a provider error.

## Where the effort belongs

Implement the bounded architecture patch and small index tests promptly. Then deliver useful kernels and publish their measured lessons incrementally. Use a compatible pinned source implementation whenever it already expresses the operator correctly. The semantic oracle must remain independent of the optimized candidate.

M01–M05 define the core investigations. M06 contains conditional deeper work, not mandatory subprojects. You have discretion over tile shapes, exact compatible library releases, sensible reduced test sizes and implementation structure. Choose those from evidence, not because a document happens to list an example first.

Do not stop at documenting that modern instructions are unavailable. Supply the actual supported alternative for the representative operator, or retain and explain a correct library/SIMT baseline when a custom path does not pay. Conversely, do not force a Tensor Core path onto a latency-/memory-bound shape merely to say it was ported.

## Source access

Use exact-path Project Control `read` for the continuing sources:

- `skills`: start with `cuda/SKILL.md`; relevant routes are listed in `sources/local-inputs.json`.
- `circuit-bending-atlas`: use the current experiment index/claim limits as well as the original architecture/mechanism cards.

Do not use natural-language/scout queries for those two sources. Ordinary exact local file reads are appropriate in the implementing environment. Keep the atlas and CUDA skill unchanged; stale links or source-label corrections are noted in the package rather than repaired through an unrelated project.

## Required judgment

Keep experimental observation, explanation and guarantee separate. The atlas and several papers contain limitations or transcription issues already adjudicated in `sources/source-adjudications.md`. Do not copy their erroneous version label, softmax equation, host ordinal assumption or numerical generalization into the wiki.

For all public performance claims, retain the native six fields plus exact `source_locator`, and the wider measurement contract in the linked report. Never attach a borrowed paper result to a local kernel. Do not claim `compiled`, `runnable`, `benchmarked`, `verified`, or “optimized” from metadata alone.

## Stop expanding when

The representative implementation is correct, a useful baseline/measurement exists, the proposed alternatives have a clear result or honest unresolved boundary, and a reader can reproduce the next relevant step. Broader shape support, backward training, another quantized format, or further profiling becomes follow-on work only when it adds value. No production serving system, all-architecture compiler or exhaustive microarchitecture study is necessary to complete this plan.

## Final handoff

Return the component commit, short query/validation receipts, kernel-family evidence and applicability limits, and a small list of worthwhile unresolved performance questions. Supply KDA with the exact validated commit and integration instructions; do not silently update another workspace. State that the native baseline corpus is useful and measured, not globally optimal.
