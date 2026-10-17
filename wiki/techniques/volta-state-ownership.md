---
id: technique-volta-state-ownership
title: Volta recurrent-state ownership across tokens
type: technique
architectures: [sm70]
tags: [kernel-fusion, gated-delta-net]
confidence: experimental
reproducibility: benchmarked
prerequisites: []
related: [kernel-volta-gated-delta]
sources: [blog-volta-recurrence-measurements, doc-volta-gdn, doc-volta-cuda-12-9]
artifact_dir: artifacts/kernels/volta-recurrence/variants
---

# Ownership choice

A recurrent update has two natural state owners. In a sequential baseline, a
kernel owns one timestep: it loads each active state row, updates it, writes it
back, then the next launch consumes that state. In a retained-state mapping, a
sequence CTA owns all its timesteps. Each row lane loads its `[d_k]` state row
once, carries it through the token loop, and writes final state once. This
trades repeated source-level state requests and launch boundaries for a longer
kernel lifetime and a larger private row.

For the gated-delta operator, state is FP32 `[sequence,d_v,d_k]`. At each live
step it applies

```cuda-cpp
float delta_dot = 0.0f;
for (int j = 0; j < DK; ++j) {
  const float decayed = alpha_step * row_state[j];
  delta_dot = fmaf(decayed, k_step[j], delta_dot);
}
const float error = v_step[row] - delta_dot;
float output = 0.0f;
for (int j = 0; j < DK; ++j) {
  const float decayed = alpha_step * row_state[j];
  row_state[j] = fmaf(beta_step * error, k_step[j], decayed);
  output = fmaf(row_state[j], q_step[j], output);
}
```

The first loop reduces `D@k` before calculating the prediction error. The
second loop applies the rank-one correction, with `output` accumulating from
the just-updated state, so the read is `S@q` rather than `S_previous@q`.

The runner assigns one CTA to each independent sequence and one active lane
to each `d_v` state row. The current token's keys, queries, values, and gates
are placed in shared memory; CTA barriers ensure all row lanes finish using
them before reuse. The retained mapping carries the entire `d_k` row for the
sequence. It does not reduce the mathematical history or alter the caller's
keys. In split execution, the returned final state is the next segment's
initial state, and outputs retain their original timestep positions.

# Source-level request model

Counting one FP32 state load and one state store for every active state step
gives the per-token path `2 * live_steps * d_v * d_k * 4` requested bytes.
The retained path reads and writes state once for each sequence with at least
one live token: `2 * nonempty_sequences * d_v * d_k * 4` requested bytes.
The fixed bundle yields:

| Case | Live steps / active sequences | Per-token source requests | Retained source requests |
|---|---:|---:|---:|
| Decode, `d_v=d_k=32` | 137 / 137 | 1,122,304 B | 1,122,304 B |
| Short prefill, `d_v=d_k=32` | 1,845 / 101 | 15,114,240 B | 827,392 B |
| Long prefill, `d_v=d_k=64` | 13,014 / 111 | 426,442,752 B | 3,637,248 B |

These numbers count nominal source-level global state requests. They are not
measured HBM transactions; cache reuse and device memory behavior are not
derived from the source formula. The largest allocated state array is
`128*64*64*4 = 2,097,152` bytes (2 MiB), below the V100's 6 MiB L2 capacity.
Capacity alone does not show that state data remained in L2 or quantify HBM
traffic.

# Measured boundary

The [bounded measurement](../../sources/blogs/volta-recurrence-measurements.md)
compares the complete per-token launch sequence with one retained sequence
launch on a Tesla V100-SXM2-16GB:

| Case | Per-token event median (µs) | Retained event median (µs) | Result |
|---|---:|---:|---|
| Decode, `T=1,B=256,d_v=d_k=32` | 19.264 | 19.456 | Overlapping seven-sample ranges; inconclusive |
| Short prefill, `T=33,B=128,d_v=d_k=32` | 292.864 | 40.768 | Retained route faster in this case |
| Long prefill, `T=257,B=128,d_v=d_k=64` | 6,993.632 | 398.048 | Retained route faster in this case |

Each timing role has two warmups and seven samples. The complete CUDA-event
interval spans all per-token launches or the one retained launch and may
include device idle while the host submits baseline launches. Reset and state
synchronization, allocation, and transfers are outside sample timing. Separate
host enqueue and completion measurements are published with the raw sample
vectors. Launch count, state ownership, synchronization, and resource behavior
all change together, so this comparison does not establish that fewer state
requests alone caused the measured gains or set a general routing threshold.

# Resource and numerical limits

The retained row is a private array, but that source-level intent alone does
not prove hardware register residency. PTXAS reported 152 registers per thread
and zero spills for the `d_k=64` candidate; the per-token kernel reported 168
registers per thread and zero spills. A single Nsight Compute sample also
reported 152 registers per thread, but occupancy was unavailable. Its mixed
classification and counter-valid/timing-invalid status make it a limited
diagnostic, not an occupancy estimate or timing result. Recheck spills and
resource behavior as dimensions and compiler versions change.

The correctness fixture uses represented FP32 keys generated as unit vectors
and gates in `[0,1]`, including exact zero and near-one values. The runner
checks key norms within the fixture tolerance and does not normalize inputs.
The independent FP64 oracle is compared per output and final-state value with
`abs(actual-reference) <= 2e-5 + 2e-4*abs(reference)`. This is a fixed test
threshold, not a general numerical error guarantee or a hardware property.

The artifact contains no chunkwise recurrence candidate. The paper discusses
chunkwise evaluation in its training approach, while this measurement uses
already resident complete inputs and exposes no chunk-planning or correction
interface. Chunking remains deferred until an equivalent bounded formulation,
real caller availability, and the full preparation/correction/conversion/carry
cost can be evaluated. The exact implementation and evidence are indexed in
[`benchmark_runs/volta-recurrence/summary.json`](../../benchmark_runs/volta-recurrence/summary.json)
and [`kernel-volta-gated-delta`](../kernels/volta-gated-delta.md).
