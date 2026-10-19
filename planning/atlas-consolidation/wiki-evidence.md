# KernelWiki evidence migration: Circuit-Bending Atlas

This note records the empirical evidence portion of the Atlas consolidation.
It was checked against Atlas documentation commit
`5b5629b001da4a42d6f41a84d94f73c07a29d478`, including `E00`–`E39`,
`CLAIM_LIMITS.md`, `INTERPRETING_RESULTS.md`, `claim_limits.json`, the portable
per-experiment JSON, `provenance.json`, and `sources/SOURCES.md`. KernelWiki
adds authored summaries and pinned locators; it does not vendor Atlas content
or local raw artifacts.

The campaign identity is `v100-20261006T145155Z-4bd01216`; the benchmark
source-code revision recorded in its provenance is
`f223c51dcfacab602e9bc68b3e65cc75730dc7f8`. The source-tree and file hashes
are `5bacf9f34b437045719772d65b47a2a4767e910373ed6413f6766d24c42ae80c` and
`318510f3097352432ccce8d38af7b1a95403d63d3acafaa8f7fd514ec2c124f2`.
Provenance identifies a Tesla V100-SXM2-16GB (`sm_70`), CUDA 12.9.86, driver
580.173.02, Nsight Compute 2025.2.1, Nsight Systems 2025.3.1, and Compute
Sanitizer 2025.2.1. These identify the historical campaign, not a portable
performance recommendation.

Portable summaries and provenance are tracked in the Atlas tree. Raw timing
vectors, profiler captures, and controller logs referenced from local paths
are unavailable in the tracked source snapshot. No numerical timing or
performance claim is recreated here. `GPU_RUN` means only that some cases
executed; proposed protocol text is not evidence that every sweep ran.
The Atlas GitHub origin is access-gated for unauthenticated readers. The
immutable URLs preserve identity; a local reader can recover a file using
`git -C <atlas-checkout> show
5b5629b001da4a42d6f41a84d94f73c07a29d478:<path>`.

## Evidence crosswalk

The original IDs below remain stable. “Campaign source” routes to the pinned
source page and its exact per-experiment summary; “Wiki page” identifies where
this migration adds a short, queryable interpretation. The source retains the
full case-level status and limitations for IDs without a dedicated Wiki page.

| ID | 2026 campaign status | Evidence topic and destination |
|---|---|---|
| E00 | CPU_ONLY | Device inventory/capability read; campaign source only. |
| E01 | CPU_ONLY | CPU algebra and encoding assertions; campaign source only. |
| E02 | GPU_RUN | Whole-kernel instruction schedule timing; campaign source only. |
| E03 | GPU_RUN | Selected cross-pipeline schedule timing; causal explanation unresolved. |
| E04 | NOT_RUN | Register-bank and operand-reuse question remains unmeasured. |
| E05 | GPU_RUN | Selected lookup/aggregation and software ordinal rank/select; no arbitrary-payload `__fns` result. |
| E06 | GPU_RUN | Shared-bank and phase cases; no causal counter evidence. |
| E07 | GPU_RUN | Access-pattern measurements; no cache-level/TLB attribution. |
| E08 | GPU_RUN | Virtual-footprint timings; no page-placement/TLB attribution. |
| E09 | GPU_RUN | [MMA coordinate observation](../../wiki/hardware/volta-atlas-e09-mma-coordinate.md); WMMA lane ownership unestablished. |
| E10 | GPU_RUN | Existing Volta numerics page plus campaign source; finite fingerprint, no general error bound. |
| E11 | GPU_RUN | [Packed intersection observation](../../wiki/hardware/volta-atlas-e11-packed-intersection.md); actual repeated-use sweep absent. |
| E12 | GPU_RUN | [Independent transform observation](../../wiki/hardware/volta-atlas-e12-independent-transform.md); no coupled recurrence. |
| E13 | GPU_RUN | Native arithmetic led three selected random-input size/reuse points; not a general crossover. |
| E14 | GPU_RUN | Exact-increment cases with implementation-estimated atomics; not hardware transaction counts. |
| E15 | GPU_RUN | One finite publication/termination protocol; no fairness or slot-reuse proof. |
| E16 | GPU_RUN | [Resident scheduling observation](../../wiki/hardware/volta-atlas-e16-resident-scheduling.md); selected regular work only. |
| E17 | GPU_RUN | Unequal-work DMA/compute cases; no equal-work overlap conclusion. |
| E18 | GPU_RUN | Selected peer paths on one physical GPU pair; no coherence conclusion. |
| E19 | GPU_RUN | Full peer signaling protocols; not primitive-only latency. |
| E20 | GPU_RUN | Pair-specific payload rates; no independent physical-link counter. |
| E21 | GPU_RUN | Page-verified node-0 transfer cases; node-1 binding was not run. |
| E22 | GPU_RUN | Tested stream-memory sequences; no advantage over event baseline established. |
| E23 | GPU_RUN | [VMM alias observation](../../wiki/hardware/volta-atlas-e23-vmm-aliasing.md); selected serial modulo baseline only. |
| E24 | GPU_RUN | Selected complete host-visible costs for four placement paths; migration/access-counter cause not established. |
| E25 | GPU_RUN | Adaptive observation cost dominated the selected synthetic phases; no online-policy win established. |
| E26 | GPU_RUN | Finite approximation corpus; no worst-case error guarantee or screen/refine implementation. |
| E27 | NOT_RUN | Restricted component-remap hypothesis; zero cases. |
| E28 | NOT_RUN | Restricted copy-engine semaphore hypothesis; zero cases. |
| E29 | NOT_RUN | Restricted QMD dependency hypothesis; zero cases. |
| E30 | NOT_RUN | Restricted binary transformation hypothesis; zero cases. |
| E31 | GPU_RUN | One finite workload and at most two MPS clients; no fairness or fault-isolation result. |
| E32 | GPU_RUN | Two fixed public launch points; does not validate E28 or E29 restricted descriptor behavior. |
| E33 | GPU_RUN | Two sequential synthetic schedules and read-only uncalibrated telemetry. |
| E34 | CPU_ONLY | One read-only RAS/platform status snapshot; no GPU workload or reliability inference. |
| E35 | CPU_ONLY | Software-visible platform inventory; no physical/electrical evidence. |
| E36 | CPU_ONLY | Toolchain/image metadata, with a compiled-only subcase; no runtime library dispatch proof. |
| E37 | GPU_RUN | Selected documented PTX BoolOp truth checks; not an edited-cubin experiment. |
| E38 | CPU_ONLY | Five offline claim checks; no silicon behavior measured. |
| E39 | GPU_RUN | [Bitplane full-adder observation](../../wiki/hardware/volta-atlas-e39-bitplane-addition.md); no crossover established. |

The campaign status counts are 29 `GPU_RUN`, six `CPU_ONLY`, and five
`NOT_RUN`. The `CPU_ONLY` IDs (E00, E01, E34, E35, E36, E38) are not GPU
evidence. E36 includes compiled metadata inspection, but not a native runtime
operator test. E27–E30 require restricted command, descriptor, register, or
binary-level control and executed zero cases. Public CUDA graph/child-launch
observations in E32 and documented PTX checks in E37 are different operations;
neither turns those restricted hypotheses into accessible CUDA mechanisms.

## Source authority and limits

The existing `blog-volta-atlas-campaign` source page remains untouched. This
new record, [`blog-atlas-v100-campaign-20261006`](../../sources/blogs/atlas-v100-campaign-20261006.md),
pins the complete evidence snapshot and campaign metadata. The six detailed
Wiki pages are `source-reported`; none is marked `verified`, and none carries
a numeric `performance_claims` entry. Hardware contract facts remain in the
existing or separately enriched architecture pages. These experiment notes
do not establish cross-workload speedups, general numerical guarantees,
causal microarchitectural explanations, public reachability of restricted
interfaces, or suitability for biological/scientific workloads.

No GPU work was run for this migration. Full raw reruns are not necessary to
preserve the bounded results, and the local traces needed to reproduce every
measurement are unavailable in the tracked Atlas source.
