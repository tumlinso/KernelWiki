# Source adjudications and evidence boundaries

These are decisions already supported by the inspected sources. Do not make the implementing agent rediscover them. Corrections are local to the KernelWiki additions; this package does not modify the continuing CUDA skill or atlas.

| Issue | Adjudication | Evidence / consequence |
|---|---|---|
| Atlas PTX version label | The CUDA 12.9.1 archive URL identifies PTX ISA **8.8**, not 9.4. | AT-SOURCES, EXT-PTX. Use the archive's actual version and exact target notes. |
| Archived “not run” versus later campaign | The original atlas entrypoints describe the original authoring session. Later experiment pages describe a measured subset. | AT-ROOT, AT-LIMITS, AT-INTERPRET. Do not treat either as proof that all 40 protocols ran. |
| Explicit MMA map versus WMMA | E09 validates one explicit row/column FP32 form. WMMA only had canonical output/finite slot checks. | AT-E09. New raw WMMA ownership claims require new evidence; preferably keep WMMA opaque. |
| SparkAttention equations | Displayed scale placement is outside softmax and the displayed normalizer formula is inconsistent. | EXT-SPARK; M03 supplies an independently derived stable operator/merge. Do not copy typographical mathematics as the oracle. |
| Accumulator precision | FP16 and FP32 accumulator paths can have different layouts and different errors. | EXT-SPARK, EXT-FASTATTN, AT-R05, EXT-FTTN. A faster path is not a like-for-like layout improvement if arithmetic changed. |
| Register-file units/SKU facts in papers | Do not propagate ambiguous “64 KB register” descriptions or one author's GPU SKU as universal V100 facts. | AT-R01 and official tuning/runtime attributes take precedence. Record actual device capacity. |
| Native snippet validation | The current validator checks code-like fences, not NVCC compilation. | KW-VALIDATE. Kernel/runtime evidence must be separate. |
| Architecture aliases | Text aliases and architecture filters are separate mechanisms in the current implementation. | KW-ALIASES, KW-QUERY, KW-POLICY. Register product/family behavior in the authoritative matching path too. |
| Language guide index | A second guide can overwrite the first under the existing single-slot data structure. | KW-INDEX. Small collection fix and one fixture, not a new index engine. |
| Generic Volta versus V100 | Family evidence is not exact SM70 applicability. Neighboring SM72 features must not leak into V100. | EXT-LEGACY, EXT-PTX, AT-R04. SM72 recognition is only an attribution boundary here. |
| CUDA versus DSL support | CuTe C++ templates, CuTe Python DSL and Triton lowering are distinct. | EXT-CUTE-DSL, EXT-TRITON, EXT-CUTLASS-MMA. A helper accepting `70` or a build succeeding does not prove a native Tensor Core backend. |
| Low-bit storage | Decoding FP8/FP4/INT4 storage is not native tensor arithmetic in those formats. | AT-R04, AT-R15, M04. Preserve decoder/scale/rounding semantics explicitly. |
| Historical microbenchmarks | Jia's measurements are useful hypotheses, not timeless latency/register-allocation constants. | EXT-JIA, AT-R02. Use focused current-binary evidence when it affects a hot kernel. |
| Atlas E06/E07/E08 explanations | Timings do not alone establish bank/cache/TLB causality. | AT-LIMITS. Design controlled changes only for the causal question the kernel actually needs. |
| Atlas E16 and E39 | No general persistence win or established bitplane arithmetic crossover is supported. | AT-E16, AT-LIMITS. Strong batching/native arithmetic baselines remain. |
| Community V100 performance | Different precision, power, operator and serving boundaries cannot be mixed into one speedup. | CAND-1CAT, CAND-1CAT-666. The inspected PR discloses unresolved quality/performance targets; it is a candidate source, not a locally reproduced benchmark. |
| Public code versus source quality | A merged PR or pinned header is not automatically safe/correct for the chosen operation. | Preserve license/provenance, inspect relevant code and validate independently. No full third-party library is bundled here. |
| NCU dependency | A usable Volta port at `~/ncu-report-skill` is a user-provided implementation prerequisite. | Do not add a port task. Inspect its actual interface and record its version; no invented Blackwell flags. |
| Historical host ordinals | Old fast-pair GPU indices are observations, not current execution authority. | SK-ROUTER, SK-HOST. Use controller-selected device UUIDs and current facts. |

## The source inventory is not a to-do list

All 42 registrations in the atlas source registry were reviewed for scope. The relevant architecture, numerical, layout and experiment material was read in depth. This does **not** claim that every external linked paper/header or every capsule was independently reaudited. The table below prevents the registry's breadth from expanding this bootstrap accidentally.

| Atlas sources | Treatment in this package |
|---|---|
| S01, S02 | Architecture/resource grounding. Official archived guide and runtime facts for canonical claims; do not import platform diagrams wholesale. |
| S03 | Core exact PTX contracts; version-label correction above. |
| S04 | Conditional operand/instruction-delivery hypotheses, not a new ISA campaign. |
| S05 | Focused binary inspection tooling when a named hot kernel needs it. |
| S06 | Compiler/architecture compatibility boundary. |
| S07 | Attention ownership/dataflow; equations and precision require care. |
| S08, S09 | Conditional integer/packed decoding helpers for M04; no general bit-arithmetic project. |
| S10 | Warp participation/synchronization guidance, also grounded in the programming guide. |
| S11 | No CuAssembler/binary-editing task. |
| S12–S18 | Driver, copy-engine, MMU, stream-memory and VMM investigations remain outside this kernel-focused extension. |
| S19 | Tensor reduction/scan is an optional candidate only if an actual M02 limiter justifies it; do not force dense tensor reformulation. |
| S20 | Core numerical stress-test design. |
| S21 | Mixed-precision recovery is a conditional source, not a required new arithmetic implementation. |
| S22, S23 | Concrete pinned SM70 MMA/layout implementation precedents; relevant code ranges inspected externally. |
| S24 | Fragment-ownership lesson may be relevant; no tcFFT port or FFT benchmark requirement. |
| S25–S29 | Communication, descriptors, faults and access-counter policy remain outside scope. |
| S30 | CUDA language, memory and synchronization contracts. |
| S31 | VMM outside scope. |
| S32 | Basic current device capability query only. |
| S33 | GPUDirect/RDMA outside scope. |
| S34 | Existing instrumentation can use CUPTI; no new CUPTI framework or sampling backend. |
| S35 | Profiling interpretation through the user-provided Volta NCU skill. |
| S36 | Read-only environment/telemetry facts where needed; no power, reset or clock campaign. |
| S37–S41 | Nouveau/firmware/MPS investigations remain in the atlas and their own projects. |
| S42 | Relevant public device attributes only; no peer/stream-control subsystem. |

Additional external inputs beyond the atlas include FastAttention, current tool/library support records, Gated Delta Networks, the online softmax source, and a pinned current SM70 community PR lead. Their inspection boundaries are in `external-sources.json`.
