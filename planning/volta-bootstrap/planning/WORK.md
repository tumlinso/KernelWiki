# Bounded work plan

Nine implementation outcomes plus one aggregate closeout. The material briefs contain the engineering decisions; this list is not an invitation to create sub-epics for every test.

## Dependency structure

`01 → 02`; `01 → 03`; `02+03 → 04,05`; `04+05 → 06,08`; `04 → 07`; `02…08 → 09 → aggregate close`.

Independent source review and implementation preparation may proceed using existing delegation, but the native initial lane is exclusive. Do not create another scheduling system.

## KWV-01 — Register Volta in the existing architecture and navigation machinery

Make SM70 exact/family/product attribution work natively with a small patch; fix diagnostics and multiple-language-guide indexing without rewriting the engine.

Dependencies: None.

### Completion evidence

- V100/sm70/sm_70 and family filters obey exact/family/unknown semantics.
- SM72 is only a neighboring attribution boundary; no false sm70a/sm700 acceptance.
- Legacy controls work; language index retains both guides.
- A focused CPU test passes; no architecture loader, ranking rewrite or full-corpus refresh is introduced.

### Delegated judgment

Use the existing repository conventions and the smallest correction that satisfies the outcome. Do not fix unrelated historical content or introduce generic abstractions without a concrete need.

References: `planning/volta-bootstrap/design/01-extension-design.md`, `planning/volta-bootstrap/design/03-validation-and-evidence.md`.

## KWV-02 — Adopt grounded Volta sources, hardware material and routing

Use the prepared drafts and source adjudications to publish a correct initial Volta knowledge layer, preserving the continuing CUDA skill and atlas.

Dependencies: KWV-01.

### Completion evidence

- Prepared source/hardware pages are adopted or mapped to exact equivalent existing records, with native IDs and valid source links.
- Confidence stays source-reported/concept unless stronger evidence actually exists.
- SKILL/primer/index route Volta independently without retagging modern-only content.
- No source corpus replacement, schema-wide migration or fabricated historical PRs.

### Delegated judgment

Use the existing repository conventions and the smallest correction that satisfies the outcome. Do not fix unrelated historical content or introduce generic abstractions without a concrete need.

References: `planning/volta-bootstrap/seed/README.md`, `planning/volta-bootstrap/sources/README.md`, `planning/volta-bootstrap/sources/source-adjudications.md`, `planning/volta-bootstrap/design/02-coverage-and-priorities.md`.

## KWV-03 — Qualify the existing native measurement path and reusable receipts

Use the CUDA controller and user-supplied Volta NCU skill to establish a small working compilation/correctness/timing/profile path. Do not build another toolkit.

Dependencies: KWV-01.

### Completion evidence

- Actual GPU UUID/SM70, compiler/sanitizer/library/NCU identities and user port commit are recorded.
- One native correctness and sanitizer smoke plus normal timing and readable selected profile have real receipts.
- Thin receipt helper rejects missing/failed/unbound evidence and does not treat not_run as evaluated_not_promoted.
- Language guides include genuine tested examples; preserve CuTe C++/Python and Triton lowering distinctions.
- No privileged changes or persistent campaigns; no duplicated profiling engine.

### Delegated judgment

Use the existing repository conventions and the smallest correction that satisfies the outcome. Do not fix unrelated historical content or introduce generic abstractions without a concrete need.

References: `planning/volta-bootstrap/materials/M00-toolchain-and-numerical-contracts.md`, `planning/volta-bootstrap/design/03-validation-and-evidence.md`.

## KWV-04 — Build and investigate native GEMM, fragment ownership and software staging

Deliver a correct useful SM70 GEMM baseline and test the actual costs of staging and retaining fragment layouts under a fixed numerical contract.

Dependencies: KWV-02, KWV-03.

### Completion evidence

- Independent small oracle, library baseline and real native compilation/correctness records exist.
- Reuse E09 exact mapping; new forms receive only the additional tests they require.
- Compare meaningful stage/layout alternatives with useful versus executed work and full operator costs.
- Record selected profile, retained implementation, losing shapes and precision limitations; no optimality or bank-cause overclaim.

### Delegated judgment

Choose the best existing source/baseline and a small set of candidate shapes from the material. A losing optimization can be `evaluated_not_promoted` only after the required correct representative implementation and the relevant measured comparison exist. An unrun/environment-blocked test does not satisfy this outcome.

References: `planning/volta-bootstrap/materials/M01-mma-layout-and-gemm.md`, `planning/volta-bootstrap/materials/M00-toolchain-and-numerical-contracts.md`, `planning/volta-bootstrap/materials/M06-targeted-deeper-investigations.md`, `planning/volta-bootstrap/design/03-validation-and-evidence.md`.

## KWV-05 — Build row kernels and locate memory/fusion crossovers

Supply stable softmax and one normalization operator; test useful traffic/request and residency tradeoffs without building a reduction library.

Dependencies: KWV-02, KWV-03.

### Completion evidence

- Stable, independently checked softmax and normalization paths support documented tail/mask/stride contracts.
- Compare selected vector/warp/CTA/fusion alternatives only where meaningful.
- Actual traffic, request/instruction costs and register/shared effects are distinguished when evidence permits.
- Published boundaries include preprocessing, full producer/consumer costs and an honest retained baseline.

### Delegated judgment

Choose the best existing source/baseline and a small set of candidate shapes from the material. A losing optimization can be `evaluated_not_promoted` only after the required correct representative implementation and the relevant measured comparison exist. An unrun/environment-blocked test does not satisfy this outcome.

References: `planning/volta-bootstrap/materials/M02-memory-row-kernels-and-fusion.md`, `planning/volta-bootstrap/materials/M00-toolchain-and-numerical-contracts.md`, `planning/volta-bootstrap/materials/M06-targeted-deeper-investigations.md`, `planning/volta-bootstrap/design/03-validation-and-evidence.md`.

## KWV-06 — Implement Volta attention dataflow for prefill and decode

Reuse matrix and row work to produce correct forward attention, then investigate fragment exchange, useful query grouping and bounded KV partitioning.

Dependencies: KWV-04, KWV-05.

### Completion evidence

- Correct scaled-dot operator, masks, independent per-head normalization, FP32 running state and any probability rounding are explicit.
- Prefill and low-query decode have real checked implementations and matched benchmarks.
- Compare a meaningful ownership/GQA/split-KV candidate selected by bottleneck evidence.
- One supplied selected-block correctness case demonstrates the sparse applicability boundary without a serving planner.
- No history/mask omission, precision shortcut or borrowed paper speedup is presented as a valid optimization.

### Delegated judgment

Choose the best existing source/baseline and a small set of candidate shapes from the material. A losing optimization can be `evaluated_not_promoted` only after the required correct representative implementation and the relevant measured comparison exist. An unrun/environment-blocked test does not satisfy this outcome.

References: `planning/volta-bootstrap/materials/M03-attention-prefill-decode-and-sparsity.md`, `planning/volta-bootstrap/materials/M00-toolchain-and-numerical-contracts.md`, `planning/volta-bootstrap/materials/M06-targeted-deeper-investigations.md`, `planning/volta-bootstrap/design/03-validation-and-evidence.md`.

## KWV-07 — Implement grouped/gated math and one exact low-bit storage path

Reuse the native matrix substrate to test grouped/gated fusion and one W4A16 decoding contract; avoid a MoE runtime or format-compatibility project.

Dependencies: KWV-04.

### Completion evidence

- Grouped/gated outputs preserve group IDs, duplicates, empty groups and numerical policy.
- One explicitly defined low-bit format has an independent decoder/oracle and all-value/tail/scale tests.
- Compare fused versus materialized decoding and small versus moderate M under the same represented weights.
- Include preparation, reuse and scheduling costs; test persistence only for a real tail/skew limiter.
- Native FP8/NVFP4/INT4 tensor support and full checkpoint compatibility are not claimed.

### Delegated judgment

Choose the best existing source/baseline and a small set of candidate shapes from the material. A losing optimization can be `evaluated_not_promoted` only after the required correct representative implementation and the relevant measured comparison exist. An unrun/environment-blocked test does not satisfy this outcome.

References: `planning/volta-bootstrap/materials/M04-grouped-gated-and-lowbit.md`, `planning/volta-bootstrap/materials/M00-toolchain-and-numerical-contracts.md`, `planning/volta-bootstrap/materials/M06-targeted-deeper-investigations.md`, `planning/volta-bootstrap/design/03-validation-and-evidence.md`.

## KWV-08 — Implement a correct recurrent state kernel and test justified chunking

Publish the gated-delta recurrence with explicit state orientation/history and assess state retention or a supported small chunk without inventing a general dynamics framework.

Dependencies: KWV-04, KWV-05.

### Completion evidence

- Independent recurrence oracle validates every output and final state, including nonzero initial state and boundary gates.
- Sequential/fused ownership comparison has source-bound correctness and normal timing evidence.
- Any chunk candidate uses a demonstrated equivalent formulation and includes setup/carry costs.
- Long-sequence and segmented-carry tests reject history truncation, misplaced decay and sequence mixing.
- A rejected chunk is recorded honestly while the correct representative baseline remains useful.

### Delegated judgment

Choose the best existing source/baseline and a small set of candidate shapes from the material. A losing optimization can be `evaluated_not_promoted` only after the required correct representative implementation and the relevant measured comparison exist. An unrun/environment-blocked test does not satisfy this outcome.

References: `planning/volta-bootstrap/materials/M05-recurrent-state-and-chunks.md`, `planning/volta-bootstrap/materials/M00-toolchain-and-numerical-contracts.md`, `planning/volta-bootstrap/materials/M06-targeted-deeper-investigations.md`, `planning/volta-bootstrap/design/03-validation-and-evidence.md`.

## KWV-09 — Publish the coherent corpus and produce the KDA handoff

Close the source/artifact/query links, run the compact regressions and native validation, and supply a tested dependency commit with explicit performance limits.

Dependencies: KWV-02, KWV-03, KWV-04, KWV-05, KWV-06, KWV-07, KWV-08.

### Completion evidence

- Generated indexes preserve existing guides and reflect the new exact/family coverage.
- Native schema/link checks and all required representative evidence checks pass, with baseline inherited issues separately reported rather than silently rewritten.
- V100 hardware/attention and legacy Hopper/Blackwell queries and source following work.
- Publication distinguishes measured, rejected and deferred performance work, with code licenses/provenance intact.
- Provide exact KernelWiki commit for KDA pinning; do not modify another workspace or claim global optimality.

### Delegated judgment

Use the existing repository conventions and the smallest correction that satisfies the outcome. Do not fix unrelated historical content or introduce generic abstractions without a concrete need.

References: `planning/volta-bootstrap/design/01-extension-design.md`, `planning/volta-bootstrap/design/02-coverage-and-priorities.md`, `planning/volta-bootstrap/design/03-validation-and-evidence.md`.

