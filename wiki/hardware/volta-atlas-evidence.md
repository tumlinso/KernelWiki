---
id: hw-volta-atlas-evidence-index
title: 'Volta: bounded Circuit-Bending Atlas campaign evidence'
type: hardware
architectures:
- sm70
tags:
- mma-sm70
confidence: source-reported
reproducibility: concept
related:
- hw-volta-mma
- hw-volta-numerics
- hw-volta-participation
- hw-volta-resources
- hw-volta-atlas-e09-mma-coordinate
- hw-volta-atlas-e11-packed-intersection
- hw-volta-atlas-e12-independent-transform
- hw-volta-atlas-e16-resident-scheduling
- hw-volta-atlas-e23-vmm-aliasing
- hw-volta-atlas-e39-bitplane-addition
sources:
- blog-atlas-v100-campaign-20261006
aliases: []
---

This page routes the bounded empirical summaries from the Circuit-Bending
Atlas V100 campaign. Its [pinned source record](../../sources/blogs/atlas-v100-campaign-20261006.md)
identifies the source revision, campaign, software and machine. The Atlas
[claim-limits index](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/CLAIM_LIMITS.md)
and [interpretation guide](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/INTERPRETING_RESULTS.md)
define how to read every status and limitation.

The Atlas GitHub origin is access-gated; these immutable URLs are unavailable
without repository access. Use the local-checkout `git show` command in the
pinned source record to retrieve the same files.

The campaign records 29 `GPU_RUN`, six `CPU_ONLY`, and five `NOT_RUN` experiment
summaries. A GPU status records that some cases ran; it does not assert that the
originally proposed protocol, cases, sweeps, or application test completed.
The CPU-only records include inventory, algebra, status/platform, compiled
metadata, and offline falsification work. Those records do not establish GPU
runtime behavior. Portable JSON summaries and provenance are tracked; raw
timing vectors, profiler captures, and controller logs referenced by local
paths are not included in the Atlas Git tree.

## Detailed bounded observations

- [E09: explicit Volta MMA coordinates](volta-atlas-e09-mma-coordinate.md)
  checks a documented explicit PTX mapping. Its WMMA subcase does not establish
  raw WMMA lane ownership.
- [E11: packed set intersections](volta-atlas-e11-packed-intersection.md)
  favors packed AND+POPC for one tested dense output contract; it did not run
  the requested reuse sweep.
- [E12: independent local transform](volta-atlas-e12-independent-transform.md)
  checks a block-diagonal Hadamard construction, not a coupled recurrent map.
- [E16: resident scheduling](volta-atlas-e16-resident-scheduling.md) found the
  resident version slower for the selected regular workloads; it makes no
  starvation or general scheduling claim.
- [E23: VMM alias view](volta-atlas-e23-vmm-aliasing.md) compares a fixed alias
  walk with a selected serial modulo baseline; it does not isolate a general
  translation or modulo effect.
- [E39: bitplane integer addition](volta-atlas-e39-bitplane-addition.md) found
  native arithmetic generally lower in the selected complete paths and no
  established bitplane crossover.

E10's finite numerical fingerprint is available alongside the existing
[Volta numerical contract](volta-numerics.md). The campaign-only summaries for
other IDs remain discoverable from the [pinned Atlas experiment directory](https://github.com/tumlinso/gpu_circuit_bending_atlas/tree/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments)
and the [portable evidence directory](https://github.com/tumlinso/gpu_circuit_bending_atlas/tree/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006).
The table gives the original card, campaign status, and this migration's
evidence disposition. “Source only” means the bounded record remains at its
pinned Atlas locator; it is not a claim that the experiment is unimportant.

| Atlas ID | Campaign status | Evidence disposition |
|---|---|---|
| E00 | CPU_ONLY | Source only: inventory and capability queries, no transfer or kernel timing. |
| E01 | CPU_ONLY | Source only: finite CPU algebra/encoding checks, not GPU validation. |
| E02 | GPU_RUN | Source only: whole-kernel schedule timing, not isolated instruction latency. |
| E03 | GPU_RUN | Source only: selected schedule contrast, cause unresolved. |
| E04 | NOT_RUN | Source only: register-bank/reuse question remains unmeasured. |
| E05 | GPU_RUN | Source only: selected lookup/aggregation and software ordinal rank/select; no arbitrary-payload `__fns` result. |
| E06 | GPU_RUN | Source only: selected shared-bank and phase cases, no causal counter proof. |
| E07 | GPU_RUN | Source only: access-pattern timing, no cache/TLB cause established. |
| E08 | GPU_RUN | Source only: virtual-footprint contrast, no page-placement or TLB cause established. |
| E09 | GPU_RUN | Detailed page; explicit PTX map checked, WMMA ownership remains unknown. |
| E10 | GPU_RUN | Existing numerics page plus source; finite test corpus, not an error theorem. |
| E11 | GPU_RUN | Detailed page; packed AND+POPC wins selected cases, reuse not swept. |
| E12 | GPU_RUN | Detailed page; block-diagonal transform, not coupled recurrence. |
| E13 | GPU_RUN | Source only: native arithmetic led three selected random-input size/reuse points; not a general crossover. |
| E14 | GPU_RUN | Source only: software-estimated atomic counts, not hardware transaction counts. |
| E15 | GPU_RUN | Source only: one finite handshake protocol, no fairness or reuse proof. |
| E16 | GPU_RUN | Detailed page; resident path slower in selected cases, no general scheduler claim. |
| E17 | GPU_RUN | Source only: unequal-work comparison; timing does not establish overlap benefit. |
| E18 | GPU_RUN | Source only: one physical pair; no coherence or topology generalization. |
| E19 | GPU_RUN | Source only: full handshake path timing, not primitive-only latency. |
| E20 | GPU_RUN | Source only: pair-specific payload rates, no independent link counter. |
| E21 | GPU_RUN | Source only: node-0 transfer cases; node-1 binding failed. |
| E22 | GPU_RUN | Source only: tested stream-memory protocol, medians overlap event baseline. |
| E23 | GPU_RUN | Detailed page; same mapped domain for requested sizes, one serial modulo baseline. |
| E24 | GPU_RUN | Source only: selected complete host-visible costs for four placement paths; migration/access-counter cause not established. |
| E25 | GPU_RUN | Source only: adaptive observation cost dominated selected synthetic phases; no online-policy win established. |
| E26 | GPU_RUN | Source only: finite approximation cases, no worst-case guarantee. |
| E27 | NOT_RUN | Restricted remap hypothesis; zero cases, no public-interface validation. |
| E28 | NOT_RUN | Restricted copy-engine actor hypothesis; zero cases, no completion-word result. |
| E29 | NOT_RUN | Restricted QMD dependency hypothesis; zero cases, no descriptor result. |
| E30 | NOT_RUN | Restricted binary transformation hypothesis; zero cases, no binary-edit result. |
| E31 | GPU_RUN | Source only: one finite workload and at most two MPS clients; no fairness or fault-isolation result. |
| E32 | GPU_RUN | Source only: two fixed public launch points; does not validate E28/E29 restricted descriptor behavior. |
| E33 | GPU_RUN | Source only: two sequential synthetic schedules, uncalibrated telemetry. |
| E34 | CPU_ONLY | Source only: read-only software-visible RAS snapshot, no reliability claim. |
| E35 | CPU_ONLY | Source only: software-visible inventory, no physical/electrical inference. |
| E36 | CPU_ONLY | Source only: metadata inspection, including compiled-only cases, no runtime dispatch proof. |
| E37 | GPU_RUN | Source only: selected documented PTX BoolOp checks, no edited-binary result. |
| E38 | CPU_ONLY | Source only: five offline claim checks, no silicon experiment. |
| E39 | GPU_RUN | Detailed page; native generally led selected full paths, no crossover established. |

In particular, E27–E30 have zero executed cases because the proposed routes
require restricted register, binary, descriptor, or command-level control.
The public CUDA graph/child-launch observations in E32 and documented PTX
truth-function checks in E37 exercise different mechanisms and do not validate
those restricted hypotheses. Treat them as unmeasured research leads, not
available CUDA capabilities.
