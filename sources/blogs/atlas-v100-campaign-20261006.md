---
id: blog-atlas-v100-campaign-20261006
title: 'Circuit-Bending Atlas: pinned V100 experiment campaign summaries'
author: Circuit-Bending Atlas project
url: https://github.com/tumlinso/gpu_circuit_bending_atlas/tree/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments
source_category: benchmark-blog
architectures:
- sm70
tags:
- mma-sm70
- wmma
retrieved_at: '2026-10-19'
description: >-
  Pinned source record for the 2026-10-06 V100 campaign's portable E00-E39
  summaries, provenance, claim limits, and interpretation guide. This record
  preserves bounded observations and non-results; it does not promote every
  originally proposed protocol to a completed experiment.
---

# Scope and provenance

This source is the Circuit-Bending Atlas documentation tree at commit
[`5b5629b001da4a42d6f41a84d94f73c07a29d478`](https://github.com/tumlinso/gpu_circuit_bending_atlas/tree/5b5629b001da4a42d6f41a84d94f73c07a29d478).
The GitHub origin is access-gated and these locators are unavailable to an
unauthenticated reader. In the local Atlas checkout, retrieve the same pinned
material with `git -C <atlas-checkout> show
5b5629b001da4a42d6f41a84d94f73c07a29d478:<path>`; for example, use
`experiments/evidence/v100-20261006/E09.json`. The URLs preserve source
identity and are not a claim of public reachability.
The experiment summaries describe campaign
`v100-20261006T145155Z-4bd01216`; its recorded benchmark-source commit is
`f223c51dcfacab602e9bc68b3e65cc75730dc7f8`. The campaign provenance records a
Tesla V100-SXM2-16GB (`sm_70`), CUDA 12.9.86, driver 580.173.02, Nsight Compute
2025.2.1, Nsight Systems 2025.3.1, and Compute Sanitizer 2025.2.1. These are
the campaign's environment, not present-day compatibility recommendations.

The tracked source tree includes portable per-experiment JSON summaries and
`provenance.json`. Raw timing vectors, profiler captures, and controller logs
referenced by local paths are not included in that tracked tree. The Wiki uses
the portable summaries and their stated boundaries; it does not imply that
those local raw artifacts are available here.

The source's [claim-limits index](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/CLAIM_LIMITS.md)
and [interpretation guide](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/INTERPRETING_RESULTS.md)
are required context for all observations. A `GPU_RUN` status means some
recorded cases ran; it does not mean the proposed sweep or mechanism was fully
tested. `CPU_ONLY`, `COMPILED_ONLY`, and `NOT_RUN` remain distinct from GPU
execution evidence.

The detailed Wiki observations in this migration are [E09](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E09.json),
[E11](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E11.json),
[E12](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E12.json),
[E16](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E16.json),
[E23](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E23.json),
and [E39](https://github.com/tumlinso/gpu_circuit_bending_atlas/blob/5b5629b001da4a42d6f41a84d94f73c07a29d478/experiments/evidence/v100-20261006/E39.json).
The [campaign index](../../wiki/hardware/volta-atlas-evidence.md) routes these
and records the status boundary for all forty cards. These Wiki pages are
source-reported observations, not new hardware qualification or performance
claims.
