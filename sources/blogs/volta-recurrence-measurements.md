---
id: blog-volta-recurrence-measurements
title: 'Volta gated-delta recurrent state ownership measurements'
author: KernelWiki Volta recurrence investigation
url: https://github.com/tumlinso/KernelWiki/blob/master/sources/blogs/volta-recurrence-measurements.md
source_category: benchmark-blog
architectures: [sm70]
tags: [gated-delta-net, kernel-fusion]
retrieved_at: '2026-10-17'
---

# Scope and recurrence contract

This experiment maps one gated-delta recurrence to two complete CUDA
implementations on a Tesla V100-SXM2-16GB (compute capability 7.0, UUID
`GPU-21131915-1488-23af-38dd-1743ae1f5cc8`). State has FP32 shape
`[sequence,d_v,d_k]`; the represented inputs are time-major and interleaved by
sequence. At every live step the implementation applies

```text
D = alpha * S_previous
e = v - D @ k
S = D + beta * outer(e, k)
o = S @ q                 # post-update state
```

With keys and values expressed as columns, this is the row-state orientation
of the gated-delta update in Equation 10 of the
[Gated Delta Networks paper](https://arxiv.org/html/2412.06464v3#S3.E10):
`S = alpha*S_previous*(I - beta*k*k^T) + beta*v*k^T`, followed by reading the
new state. The CUDA operator consumes its inputs unchanged. It does not
normalize keys, truncate history, or reset state at a segment boundary.

The immutable represented-input bundle contains three cases. Active test keys
are generated as unit vectors and checked within a fixture tolerance; they are
not changed by the runner. The per-output admission rule was fixed before
native execution as
`abs(actual-reference) <= 2e-5 + 2e-4*abs(reference)`, with an independent
FP64 recurrence over exact stored FP32 inputs.

| Case | Shape `(T,B,d_v,d_k)` | Live steps | Nonempty sequences |
|---|---:|---:|---:|
| Decode | `(1,256,32,32)` | 137 | 137 |
| Short prefill | `(33,128,32,32)` | 1,845 | 101 |
| Long prefill | `(257,128,64,64)` | 13,014 | 111 |

Input file SHA-256 is
`e760eda922a42c0d6f6338e8cc0d0b0c7971c4205c956047a55791191fced4b5` (56,285,400
bytes); payload SHA-256 is
`7c26087a2d035d692c90167cfc493756a3954cee84664a92c7ece2641611da80`. The
case contract, exact input identity, and independent oracle are bound in the
[benchmark summary](../../benchmark_runs/volta-recurrence/summary.json) and
[source closure](../../benchmark_runs/volta-recurrence/evidence/v1/source-closure/source-closure.json).

# Correctness results

Both the `per_token` baseline and `retained` candidate passed all 18 named
checks across the three cases with zero failed or non-finite elements. Each
implementation checked the full sequence and an equivalent sequence split at
available boundaries with state carried between launches. Every live output,
every final-state value, and every padded-output sentinel was checked against
the independent oracle. Per full or segmented evaluation this covers 896,320
live output values and 917,504 final-state values. The candidate's maximum
scaled output error was 0.004722 of the declared tolerance. The long case
includes nonzero initial state, zero and near-one gates, repeated and
orthogonal keys, ragged lengths, and long history. The decode case has one
input token and does not batch unobserved future inputs.

Baseline and candidate native result captures are
[baseline correctness](../../benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_correctness_baseline/foreground.stdout.txt)
and
[candidate correctness](../../benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_correctness_candidate/foreground.stdout.txt).
The fixed rule and stored oracle are detailed in
[`cpu-oracle.json`](../../benchmark_runs/volta-recurrence/cpu-oracle.json).

# Mappings and timing method

The baseline launches a separate kernel for each token. Each active sequence
CTA loads its state rows, updates them, and writes them back before the next
token launch. The candidate launches one kernel for a sequence and keeps each
complete `[d_k]` state row in the row-owning lane's private array until final
writeback. Both return every live output and the final state. The comparison
includes each route's complete launch sequence.

The timing roles used two warmups and seven samples. State reset and its stream
synchronization finish before timing; bundle I/O, allocation, input transfers,
output checking, and profile replay are outside samples. Each report separates
host enqueue, host completion, and the CUDA-event interval. The event interval
covers all T baseline launches or the single retained launch. It may include
device idle while the host submits per-token kernels and is not the sum of
kernel durations.

| Case | Per-token event median [min, max] (µs) | Retained event median [min, max] (µs) | Samples | Comparison |
|---|---:|---:|---:|---|
| Decode | 19.264 [18.880, 20.160] | 19.456 [19.008, 21.536] | 7 each | Overlapping ranges; inconclusive |
| Short prefill | 292.864 [292.608, 293.632] | 40.768 [39.968, 42.752] | 7 each | 7.18× lower retained median |
| Long prefill | 6,993.632 [6,365.856, 6,994.528] | 398.048 [397.024, 399.136] | 7 each | 17.57× lower retained median |

The long baseline's broad range is kept as measured; no samples were dropped.
All host timing medians (enqueue/completion, µs) are decode `5.726/26.280`
versus `5.572/26.074`, short prefill `72.648/300.030` versus `5.226/47.193`,
and long prefill `535.498/7001.124` versus `6.240/404.865`. Full seven-sample
vectors are available in the [baseline timing capture](../../benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_baseline_timing/foreground.stdout.txt)
and [candidate timing capture](../../benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_candidate_timing/foreground.stdout.txt).

The short and long prefill results favor retained state for these exact
whole-operator cases. Decode's approximately one-percent median difference is
smaller than the sample spread and is not evidence of a regression. These
results combine launch count, state ownership, synchronization, and resource
behavior. They do not isolate state traffic as the cause and do not establish
a dispatch crossover.

# State-request model and device diagnostics

The source-level nominal state-request model counts one load plus one store
for each active per-token state step, and one initial load plus one final store
for each nonempty retained sequence. It counts global load/store request bytes
from the source mapping, not measured HBM traffic:

| Case | Per-token nominal requests | Retained nominal requests |
|---|---:|---:|
| Decode | 1,122,304 B | 1,122,304 B |
| Short prefill | 15,114,240 B | 827,392 B |
| Long prefill | 426,442,752 B | 3,637,248 B |

For example, the long retained allocation holds `128*64*64*4 = 2,097,152`
bytes of state, below the V100's 6 MiB L2 capacity. This size comparison does
not establish cache residency or HBM transaction counts. The nominal request
reduction is a property of the source ownership mapping; the observed timing
is for the full implementations.

The sole first-build ELF was reused across roles (SHA-256
`8e6cb928f4925cfbb8470c784eb5f4498ffd4622e3b62a1c57b0d7725e75bef1`). PTXAS
reported 152 registers/thread and zero spills for retained `d_k=64`; the
per-token `d_k=64` kernel reported 168 registers/thread and zero spills. These
are compiler resource reports, not a register-residency or occupancy result.
The first correctness role built it with NVCC 12.9, V12.9.86, using
`-O3 -std=c++17 -lineinfo -arch=sm_70 -Xptxas=-v -Xcompiler=-Wall,-Wextra
-Xcompiler=-fno-omit-frame-pointer`; the exact command and result are in that
role's [`role-manifest.json`](../../benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_correctness_baseline/role-manifest.json),
with compiler resource output in `nvcc.stderr.txt`. The selected profile used
Nsight Compute CLI 2025.2.1.0, recorded in the profile role receipt and NCU
shim log. One Nsight Compute sample independently reported 152 registers per
thread, 4.50% DRAM and 8.95% SM throughput. Its class was mixed, occupancy was
unavailable, counters were valid, replay timing was invalid, and more data was
requested. Treat it as a limited kernel diagnostic; it does not explain the
unprofiled timing delta.

The candidate native application passed its checks under Compute Sanitizer,
and the preserved raw log says `ERROR SUMMARY: 0 errors`. The sanitizer wrapper
remains `partial` with `conclusive=false` and zero classified errors. This is
not described as a conclusive sanitizer pass. See the
[raw log](../../benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_sanitizer_candidate/sanitizer/run/raw.log)
and [wrapper summary](../../benchmark_runs/volta-recurrence/evidence/v1/roles/recurrence_sanitizer_candidate/sanitizer/run/summary.json).

# Chunking boundary and reproducibility

No chunk candidate was implemented or timed. The article also describes
chunkwise evaluation, but the measured artifact only retains a full sequence
or explicitly carries state across split launches. The prepared caller
contract supplies complete resident inputs and has no chunk-planning or
correction interface. Chunking remains deferred pending an equivalent bounded
formulation, evidence that a real caller already provides its inputs, and a
measurement that includes preparation, correction, conversion, and carry
costs.

The artifact source, FP64 oracle, controller role specifications, six
hash-bound receipts, build log, ELF, and profile files are indexed in
[`benchmark_runs/volta-recurrence/summary.json`](../../benchmark_runs/volta-recurrence/summary.json).
The exact runner and kernel are under
[`artifacts/kernels/volta-recurrence/variants`](../../artifacts/kernels/volta-recurrence/variants).
