---
id: kernel-volta-gated-delta
title: Volta Gated-Delta Recurrent State Kernels
type: kernel
architectures: [sm70]
tags: [gated-delta-net, kernel-fusion]
confidence: experimental
reproducibility: benchmarked
kernel_types: [gated-delta-net]
languages: [cuda-cpp]
related: [technique-volta-state-ownership]
sources: [blog-volta-recurrence-measurements, doc-volta-gdn, doc-volta-cuda-12-9]
performance_claims:
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP32 state and FP32 inputs / outputs
    shape: short prefill T=33 B=128 d_v=32 d_k=32; one per-token launch per step
    metric: CUDA-event median complete-sequence interval (us)
    value: 292.864
    source_id: blog-volta-recurrence-measurements
    source_locator: "benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_baseline_timing/foreground.stdout.txt; cases[short_prefill_many_sequences].timing.device_event_interval_median_us; seven samples in timing.device_event_interval_us"
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP32 state and FP32 inputs / outputs
    shape: short prefill T=33 B=128 d_v=32 d_k=32; fused retained state
    metric: CUDA-event median complete-sequence interval (us)
    value: 40.768
    source_id: blog-volta-recurrence-measurements
    source_locator: "benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_candidate_timing/foreground.stdout.txt; cases[short_prefill_many_sequences].timing.device_event_interval_median_us; seven samples in timing.device_event_interval_us"
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP32 state and FP32 inputs / outputs
    shape: long prefill T=257 B=128 d_v=64 d_k=64; one per-token launch per step
    metric: CUDA-event median complete-sequence interval (us)
    value: 6993.632
    source_id: blog-volta-recurrence-measurements
    source_locator: "benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_baseline_timing/foreground.stdout.txt; cases[long_prefill_many_sequences].timing.device_event_interval_median_us; seven samples in timing.device_event_interval_us"
  - gpu: Tesla V100-SXM2-16GB
    dtype: FP32 state and FP32 inputs / outputs
    shape: long prefill T=257 B=128 d_v=64 d_k=64; fused retained state
    metric: CUDA-event median complete-sequence interval (us)
    value: 398.048
    source_id: blog-volta-recurrence-measurements
    source_locator: "benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_candidate_timing/foreground.stdout.txt; cases[long_prefill_many_sequences].timing.device_event_interval_median_us; seven samples in timing.device_event_interval_us"
artifact_dir: artifacts/kernels/volta-recurrence/variants
---

# Operator contract

This artifact implements the gated-delta recurrent update in FP32 with state
layout `[sequence,d_v,d_k]`. At each active token it computes

```text
D = alpha * S_previous
e = v - D @ k
S = D + beta * outer(e, k)
o = S @ q                 # post-update state
```

Expanding the rank-one update gives
`S = alpha*S_previous*(I - beta*k*k^T) + beta*v*k^T`, which is the row-state
orientation of Equation 10 in the
[Gated Delta Networks paper](https://arxiv.org/html/2412.06464v3#S3.E10).
The kernel returns the output from the updated state. It consumes caller
inputs unchanged: there is no key normalization, history truncation, or
implicit state reset at sequence boundaries.

# Two complete mappings

[`gdn_recurrent_state.cuh`](../../artifacts/kernels/volta-recurrence/variants/gdn_recurrent_state.cuh)
contains the per-token baseline and retained-state candidate. One CTA owns one
sequence; each active lane owns one `d_v` row. The baseline enqueues a separate
kernel for every timestep, so each active step loads and stores the state row
across a kernel boundary. The candidate launches once for the sequence and
keeps its complete `d_k` row in a private array until it writes the final
state. Both mappings preserve the full recurrence and produce each live
output.

The candidate's core ownership is visible in this abbreviated source excerpt;
the full loop and synchronization are in the linked header:

```cuda-cpp
float row_state[DK];
if (row < DV) {
  for (int j = 0; j < DK; ++j) row_state[j] = states[state_base + j];
}
for (int t = 0; t < sequence_length && t < T; ++t) {
  // shared current-token vectors are loaded and consumed by every row
  // lane; CTA barriers guard shared-vector reuse between tokens.
  if (row < DV) {
    // Apply the FP32 update to row_state and write the post-update output.
  }
}
if (row < DV) {
  for (int j = 0; j < DK; ++j) states[state_base + j] = row_state[j];
}
```

This is a whole-operator comparison. Its timings include the baseline's T
launches and their host submission gaps, while the candidate uses one
sequence launch. State movement, launch behavior, synchronization, and device
resource use change together; the benchmark does not isolate state traffic
as a cause.

# Correctness and measured scope

The immutable bundle covers decode `(T=1,B=256,d_v=d_k=32)`, short prefill
`(33,128,32,32)`, and long prefill `(257,128,64,64)`. It includes ragged and
zero-length sequences, nonzero initial state, exact zero and near-one gates,
repeated and orthogonal keys, and represented unit-key fixtures. The
independent oracle uses FP64 arithmetic over the stored FP32 values. The
predeclared tolerance is
`abs(actual-reference) <= 2e-5 + 2e-4*abs(reference)` for every live output and
final-state value.

Both native variants passed all correctness, padding, and segmented-carry
checks. Segmented runs carry the returned state between launches and compare
every output and final state to the same oracle; no history is dropped. The
complete input identity, check counts, controller receipts, and timing samples
are in the [measurement record](../../sources/blogs/volta-recurrence-measurements.md)
and [benchmark summary](../../benchmark_runs/volta-recurrence/summary.json).

On the tested V100, retained state reduced the short-prefill event median from
`292.864` to `40.768 µs` and the long-prefill median from `6,993.632` to
`398.048 µs`. The one-token decode medians, `19.264` and `19.456 µs`, have
overlapping ranges and are inconclusive. These are seven-sample unprofiled
CUDA-event intervals over the complete operator; the baseline interval can
include time while the host submits its per-token launches. They establish
only the measured shapes and do not provide a general crossover or explain
the cause of the difference.

# Resource and validation limits

PTXAS reported 152 registers per thread and zero spills for retained
`d_k=64`; its per-token counterpart used 168 registers per thread with zero
spills. One Nsight Compute sample also reported 152 registers per thread, but
occupancy was unavailable. The profile was classified mixed with valid
counters and invalid replay timing. These diagnostics do not establish
occupancy or explain the unprofiled latency result. The retained private row
must be rechecked for spills and resource effects when dimensions or compiler
change.

Compute Sanitizer's raw log reports zero errors, and the native application
passed its checks. The wrapper result remains partial and nonconclusive; this
record does not claim a conclusive sanitizer pass. No chunked implementation
is present or measured. Chunking remains deferred pending equivalent
formulation and caller-availability evidence, together with complete
preparation and carry costs.
