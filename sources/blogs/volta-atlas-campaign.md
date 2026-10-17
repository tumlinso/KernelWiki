---
id: blog-volta-atlas-campaign
title: 'V100 atlas campaign: bounded ownership and numerical evidence'
author: Circuit-Bending Atlas project
url: https://github.com/tumlinso/gpu_circuit_bending_atlas
source_category: benchmark-blog
architectures:
- sm70
tags:
- mma-sm70
- wmma
retrieved_at: '2026-10-17'
---

# Evidence locator and boundary

Project Control workspace `circuit-bending-atlas`, `experiments/E09.md`, `experiments/E10.md`, `experiments/CLAIM_LIMITS.md`, and `experiments/INTERPRETING_RESULTS.md` were read directly. Their content identities are preserved in the bootstrap source ledger. The repository URL is a locator, not an immutable pin for the later documentation updates.

The reviewed E09 summary reports 256 checked coordinates for the explicit row/column FP32-accumulator `m8n8k4` form. Its WMMA case checked canonical output and finite slots, not raw lane ownership. E10 is a finite numerical fingerprint, not a general error theorem.

The campaign is `v100-20261006T145155Z-4bd01216`, with benchmark source commit `f223c51dcfacab602e9bc68b3e65cc75730dc7f8`. That code commit is not claimed to be the commit of every later summary. Portable evidence resides under `experiments/evidence/v100-20261006/`; raw traces remain local. No campaign was rerun during bootstrap preparation.

Retain each experiment's tested subset, oracle, limitations and measurement boundary when creating a performance claim. This source record supplies bounded source-reported evidence only; it does not satisfy an upstream-code confidence gate by itself.
