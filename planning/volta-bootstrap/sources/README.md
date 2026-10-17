# Source ledger

Bracketed IDs in the design/material briefs refer to the entries below. Local inputs use Project Control workspace and exact path; hashes and inspection scope are in `local-inputs.json`. External URLs and reviewed boundaries are in `external-sources.json`.

## External evidence and candidates

### EXT-VTG — NVIDIA Volta Tuning Guide (CUDA 12.9.1 archive)

https://docs.nvidia.com/cuda/archive/12.9.1/volta-tuning-guide/index.html

Architecture, software staging and resource constraints; sections 1.4.1–1.4.3.
Inspection: relevant content inspected.

### EXT-PTX — NVIDIA PTX ISA 8.8 (CUDA 12.9.1 archive)

https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html

Exact instruction target/fragment/participation contracts; do not relabel as PTX 9.4.
Inspection: relevant content inspected.

### EXT-PROG — NVIDIA CUDA C++ Programming Guide 12.9.1

https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html

WMMA, synchronization, feature boundaries and runtime resource attributes.
Inspection: relevant content inspected.

### EXT-LEGACY — NVIDIA legacy CUDA compute capability list

https://developer.nvidia.com/cuda/gpus/legacy

V100 exact capability 7.0 versus Xavier 7.2; family is not exact target.
Inspection: relevant content inspected.

### EXT-CUDA-SUPPORT — NVIDIA: Navigating GPU Architecture Support

https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/

CUDA 13 native compilation boundary and continued use of compatible earlier toolkits.
Inspection: relevant content inspected.

### EXT-NCU — Nsight Compute 2025.2 Profiling Guide

https://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html

Profiling/replay/metric interpretation; redirects to NVIDIA archive. Interface supplied by user port.
Inspection: relevant content inspected.

### EXT-CUBLAS — NVIDIA cuBLAS 12.9.1

https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html

Supported GEMM/algorithm numerical contracts; query actual cuBLASLt choices.
Inspection: relevant content inspected.

### EXT-CUSPARSE — NVIDIA cuSPARSE 12.9.1

https://docs.nvidia.com/cuda/archive/12.9.1/cusparse/index.html

Blocked-ELL SpMM type/capability and operation restrictions, section 6.6.8.
Inspection: relevant content inspected.

### EXT-CUTE-DSL — NVIDIA CuTe Python DSL FAQ

https://docs.nvidia.com/cutlass/media/docs/pythonDSL/faqs.html

Distinguish Python DSL documented supported architectures from CuTe C++ implementation templates.
Inspection: relevant content inspected.

### EXT-SPARK — SparkAttention

https://arxiv.org/html/2502.12784v1

Sections 3.2.1–3.2.2: online state and Volta fragment conversion; numerical/equation caveats recorded.
Inspection: relevant content inspected.

### EXT-FASTATTN — FastAttention

https://arxiv.org/html/2410.16663v1

Section 4.3 and Appendix B: Volta data layouts and different accumulator paths.
Inspection: relevant content inspected.

### EXT-FTTN — FTTN: Feature-Targeted Testing for Numerical Properties of Matrix Accelerators

https://arxiv.org/html/2403.00232v1

Adversarial numerical feature selection; measured properties not ISA-wide guarantees.
Inspection: relevant content inspected.

### EXT-SOFTMAX — Online normalizer calculation for softmax

https://arxiv.org/abs/1805.02867

Primary source for online normalization; this package derives the merge independently.
Inspection: abstract/metadata and relevant algorithmic framing inspected.

### EXT-GDN — Gated Delta Networks: Improving Mamba2 with Delta Rule

https://arxiv.org/html/2412.06464v3

Equation 10 and chunkwise formulation; exact state orientation/admission must be fixed in implementation.
Inspection: relevant content inspected.

### EXT-JIA — Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking

https://arxiv.org/pdf/1804.06826

Sections on registers/operand reuse/instruction delivery; register-bank page screenshot inspected. Historical compiler measurements, not current constants.
Inspection: targeted text and register-bank PDF page visually inspected.

### EXT-CUTLASS-MMA — CUTLASS Sm70 MMA implementation

https://github.com/NVIDIA/cutlass/blob/v2.11.0/include/cutlass/arch/mma_sm70.h

Exact supported m8n8k4 implementation precedent. Relevant lines 45–175 fetched via GitHub.
Inspection: relevant content inspected.

### EXT-CUTLASS-LAYOUT — CUTLASS Volta multiplicand layouts

https://github.com/NVIDIA/cutlass/blob/v2.11.0/include/cutlass/layout/tensor_op_multiplicand_sm70.h

Congruous layout/addressing implementation; relevant lines 45–205 fetched via GitHub.
Inspection: relevant content inspected.

### EXT-TRITON — Triton PR 5066: deprecate MMA-v1 / fallback to FMA below CC80

https://github.com/triton-lang/triton/pull/5066

Exact compiler lowering matters; neither version number nor successful compilation proves a Volta Tensor Core path.
Inspection: relevant content inspected.

### EXT-FT — NVIDIA FasterTransformer

https://github.com/NVIDIA/FasterTransformer

Candidate source of established FP16-era kernels; extract only an appropriate pinned operator.
Inspection: repository documentation inspected; specific implementation validation delegated.

### CAND-1CAT — 1Cat-vLLM SM70 engineering fork

https://github.com/1CatAI/1Cat-vLLM

Candidate work on GQA packing, vectorized KV loads and software quantization. Do not use serving headlines as kernel evidence.
Inspection: README/reported contracts inspected; source-level qualification required.

### CAND-1CAT-666 — 1Cat-vLLM PR 666: local head/projection layouts and precision repairs

https://github.com/1CatAI/1Cat-vLLM/pull/666

Pinned candidate starting point; description exposes precision, scale-order and quality limits. Do not import unrelated serving changes.
Inspection: GitHub PR metadata/description inspected; not the complete 51-file diff.

## Local inputs

| ID | Workspace / exact path | Role |
|---|---|---|
| KW-README | `kernelwiki` / `README.md` | Native layout, scope, maintenance and quality gates. |
| KW-SKILL | `kernelwiki` / `SKILL.md` | Blackwell/Hopper activation; native output contract. |
| KW-TAGS | `kernelwiki` / `data/tags.yaml` | Controlled vocabulary currently lacks Volta/SM70. |
| KW-SCHEMA | `kernelwiki` / `data/schemas.yaml` | Page schemas, source categories, asset and version-claim contracts. |
| KW-ALIASES | `kernelwiki` / `data/aliases.yaml` | Keyword/tag aliases distinct from architecture policy. |
| KW-POLICY | `kernelwiki` / `scripts/pr_policy.py` | Family/exact/product/guard recognition; architecture_matches_filter; relevant source ranges. |
| KW-QUERY | `kernelwiki` / `scripts/query.py` | Recursive retrieval, scoring, filters, hard-coded exclusion diagnostics. |
| KW-INDEX | `kernelwiki` / `scripts/generate-indices.py` | Recursive discovery, architecture index, single language-guide slot; relevant ranges. |
| KW-VALIDATE | `kernelwiki` / `scripts/validate.py` | Hopper relevance check, code-fence heuristic, provenance/performance/evidence checks; relevant ranges. |
| KW-REFRESH | `kernelwiki` / `scripts/refresh_candidate_ledger.py` | Bounded PR search is not an exhaustive historical source inventory. |
| KW-PRIMER | `kernelwiki` / `references/primer.md` | Existing feature, technique, kernel and language coverage. |
| SK-ROOT | `skills` / `cuda/SKILL.md` | Current CUDA routing and execution controls. |
| SK-ROUTER | `skills` / `cuda/references/architectures/volta/router.md` | Volta routes; historical host ordinals are not placement authority. |
| SK-TENSOR | `skills` / `cuda/references/architectures/volta/tensor-cores.md` | Library/TC eligibility, padding and custom ownership tradeoffs. |
| SK-LOW-TENSOR | `skills` / `cuda/references/architectures/volta/tensor-core-low-level.md` | Justification for owning a native fused/blocked tensor path. |
| SK-FUSION | `skills` / `cuda/references/architectures/volta/fusion-and-specialization.md` | HBM pass elimination, splitting and launch/capture distinctions. |
| SK-REG | `skills` / `cuda/references/architectures/volta/register-pressure-and-occupancy.md` | Registers, spills and useful occupancy; compiler summaries first. |
| SK-PROFILE | `skills` / `cuda/references/architectures/volta/profiling-interpretation.md` | Summary-first interpretation and replay/timing distinction. |
| SK-OPT | `skills` / `cuda/references/architectures/volta/optimization-guide.md` | Native compiler and implementation guidance; samples are not blanket support evidence. |
| SK-MACHINE | `skills` / `cuda/references/common/machine-aligned-design.md` | Semantic unit, ownership, complete cost and falsifiable representation choices. |
| SK-HOST | `skills` / `cuda/references/execution/host-execution.md` | Mandatory shared-host controller, toolkit, admission and foreground controls. |
| SK-BENCH | `skills` / `cuda/references/profiling/benchmark-standardization.md` | Reuse established run/summary conventions; adapt only applicable kernel scope. |
| AT-ROOT | `circuit-bending-atlas` / `README.md` | Edition, later campaign and evidence access boundaries. |
| AT-FIELD | `circuit-bending-atlas` / `FIELD_GUIDE.md` | Representation-first mechanisms; only kernel-relevant routes imported. |
| AT-SOURCES | `circuit-bending-atlas` / `sources/SOURCES.md` | All 42 source registrations reviewed for scope, not all external documents independently reaudited. |
| AT-R01 | `circuit-bending-atlas` / `reference/R01.md` | Resource/ownership hierarchy; distinguish SKU from architecture. |
| AT-R02 | `circuit-bending-atlas` / `reference/R02.md` | Dependency/operand graphs and historical microarchitecture observations. |
| AT-R03 | `circuit-bending-atlas` / `reference/R03.md` | Useful bytes versus traffic/requests and producer-consumer layouts. |
| AT-R04 | `circuit-bending-atlas` / `reference/R04.md` | Intrinsic/PTX/SASS distinctions and target boundaries. |
| AT-R05 | `circuit-bending-atlas` / `reference/R05.md` | Explicit SM70 fragment geometry versus opaque WMMA. |
| AT-R06 | `circuit-bending-atlas` / `reference/R06.md` | Participation, visibility and progress. |
| AT-R11 | `circuit-bending-atlas` / `reference/R11.md` | Complete-cost lower bounds and reuse break-even. |
| AT-R14 | `circuit-bending-atlas` / `reference/R14.md` | Shared-resource constraints in overlapped execution. |
| AT-R15 | `circuit-bending-atlas` / `reference/R15.md` | Numerical representation contracts and limits of empirical bounds. |
| AT-M24 | `circuit-bending-atlas` / `mechanisms/M24.md` | Four independent native products are a substrate, not an arbitrary coupled operator. |
| AT-M25 | `circuit-bending-atlas` / `mechanisms/M25.md` | Native fragment ownership and intermediate movement. |
| AT-LIMITS | `circuit-bending-atlas` / `experiments/CLAIM_LIMITS.md` | Case-by-case nonclaims for the 2026-10-06 campaign. |
| AT-INTERPRET | `circuit-bending-atlas` / `experiments/INTERPRETING_RESULTS.md` | Observation versus explanation; timed boundaries and evidence grades. |
| AT-E09 | `circuit-bending-atlas` / `experiments/E09.md` | 256 explicit PTX coordinate checks; WMMA canonical output only. |
| AT-E10 | `circuit-bending-atlas` / `experiments/E10.md` | Finite numerical fingerprint, not general arithmetic certification. |
| AT-E06 | `circuit-bending-atlas` / `experiments/E06.md` | Shared/shuffle semantic controls; no causal bank-counter claim. |
| AT-E16 | `circuit-bending-atlas` / `experiments/E16.md` | Resident execution lost to batching/graphs on the selected sample. |
| PC-PLAN | `project-control` / `src/todo_orchestrator/plan.py` | Native schema versions, tasks/runs/lanes/dependencies/path validation and upsert; relevant ranges. |
| PC-GATES | `project-control` / `src/todo_orchestrator/gates.py` | Native command/task_state/manual gates and CUDA admission fields; relevant ranges. |
| PC-CLI | `project-control` / `tests/test_cli_profiles.py` | Plan validate/apply --project --file grammar; unrelated secret fixture redacted by provider. |
| PC-APPLY | `project-control` / `tests/test_plan_apply.py` | Native nonmutating validation, prospective change and transactional identity checks. |
