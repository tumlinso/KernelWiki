# KWV-08 recurrent state ownership

This package contains a controller-backed SM70 comparison between a sequential
per-token recurrence and a fused sequence kernel that retains the recurrent
state rows across tokens. The measured result retains the fused route for the
tested short and long prefill cases. Decode is inconclusive: both routes have
overlapping seven-sample event ranges. This is a bounded result for the fixed
bundle and Tesla V100-SXM2-16GB, not a general dispatch rule.

## Contract and artifacts

The state is FP32, row-major `[sequence,d_v,d_k]`. Inputs are time-major and
interleaved by sequence. For every live token:

```text
D = alpha * S_previous
e = v - D @ k
S = D + beta * outer(e, k)
o = S @ q                 # post-update state
```

The kernel consumes represented inputs unchanged and does not normalize keys,
truncate history, or restart at segment boundaries. The immutable input bundle
contains decode, short-prefill, and long-prefill cases, ragged lengths, nonzero
initial states, exact zero and near-one gates, and FP64 references for every
output and final state. Input file SHA-256 is
`e760eda922a42c0d6f6338e8cc0d0b0c7971c4205c956047a55791191fced4b5` (56,285,400
bytes); payload SHA-256 is
`7c26087a2d035d692c90167cfc493756a3954cee84664a92c7ece2641611da80`.

The predeclared per-element rule is
`abs(actual-reference) <= 2e-5 + 2e-4*abs(reference)`. Both native correctness
variants passed all 18 checks over three cases, including every live output,
untouched padded output, final state, segmented-carry output, segmented
padding, and segmented final state. Across the bundle, each full and segmented
evaluation checks 896,320 live outputs and 917,504 final-state values per
variant; all failed and non-finite counts are zero. The maximum scaled output
error was 0.004722 of the allowed tolerance. The native reader verifies the
represented keys are unit length within its fixture tolerance; the kernel
does not renormalize them.

## Timing result

Each timing role runs two warmups and seven samples. State reset and its stream
synchronization finish before a sample begins. Allocation, input transfers,
output checking, and profiling are outside the timed interval. The per-token
baseline submits T separate kernels and includes its complete host enqueue
sequence. The retained route submits one fused sequence kernel. Host enqueue,
host completion, and CUDA-event interval are distinct measurements. The event
interval spans the complete launch sequence and may include device idle time
while the host submits per-token launches; it is not the sum of isolated kernel
times. Full seven-sample vectors are preserved in the role stdout captures.

| Case (T × B, d_v × d_k) | Per-token event median [range] µs | Retained event median [range] µs | Readout |
|---|---:|---:|---|
| Decode (1 × 256, 32 × 32) | 19.264 [18.880, 20.160] | 19.456 [19.008, 21.536] | Ranges overlap; no clear difference |
| Short prefill (33 × 128, 32 × 32) | 292.864 [292.608, 293.632] | 40.768 [39.968, 42.752] | 7.18× lower retained median |
| Long prefill (257 × 128, 64 × 64) | 6,993.632 [6,365.856, 6,994.528] | 398.048 [397.024, 399.136] | 17.57× lower retained median |

Host enqueue/completion medians (µs) were decode `5.726/26.280` per-token and
`5.572/26.074` retained; short prefill `72.648/300.030` and
`5.226/47.193`; long prefill `535.498/7001.124` and `6.240/404.865`.
The per-token event interval includes submission gaps. The speedups combine
the launch boundary, state ownership, and resource behavior of these two
complete mappings. They do not isolate state traffic as the cause.

The source requests one state load and one state store per live step in the
per-token path, and one initial load plus final store for each nonempty
sequence in the retained path. These are nominal source-level global-memory
requests, not HBM bytes. Exact counts and caveats are in the measurement note.
The largest allocated state array is 2 MiB, below the V100's 6 MiB L2 capacity;
that capacity comparison does not establish where requests were served.

## Correctness and diagnostics

The first per-token correctness role built the only ELF; every later role used
that exact executable (SHA-256
`8e6cb928f4925cfbb8470c784eb5f4498ffd4622e3b62a1c57b0d7725e75bef1`). PTXAS
reported 152 registers/thread and zero spills for retained `d_k=64`, and 168
registers/thread and zero spills for the per-token step. These compiler reports
do not establish occupancy. NVCC 12.9, V12.9.86 built the ELF for `sm_70` with
`-O3 -std=c++17 -lineinfo -arch=sm_70 -Xptxas=-v -Xcompiler=-Wall,-Wextra -Xcompiler=-fno-omit-frame-pointer`;
the exact command and result are in the first correctness role's
[`role-manifest.json`](evidence/v1/roles/recurrence_correctness_baseline/role-manifest.json).
The selected Nsight Compute CLI was version 2025.2.1.0. One sample independently reported
152 registers/thread, 4.50% DRAM and 8.95% SM throughput; classification was
mixed, occupancy unavailable, counter status valid, replay timing invalid,
and more data requested. This is a limited diagnostic, not timing evidence or
a causal account of the speedup.

The native candidate application passed the Compute Sanitizer invocation and
the raw log says `ERROR SUMMARY: 0 errors`, but the wrapper summary is
`partial` and `conclusive=false`. It is not recorded as a conclusive sanitizer
pass.

## Reproduction and evidence

The canonical summary links every controller receipt, lease, native output,
archive manifest, and profile/sanitizer artifact:
[`summary.json`](summary.json). Its six roles are baseline correctness,
candidate correctness, candidate sanitizer, baseline timing, candidate timing,
and one selected candidate profile. All roles used the explicit V100 UUID
`GPU-21131915-1488-23af-38dd-1743ae1f5cc8`, compute capability 7.0, and the
same immutable bundle and ELF. The frozen 39-input closure is
[`source-closure.json`](evidence/v1/source-closure/source-closure.json),
SHA-256 `fecce508e9e8e80567379b52b8a6bb88ea7c0a0f15222f711edd60a0e9598543`.
Two full Project Control source-snapshot trees are also preserved in the local
evidence directory, but are not part of the portable committed package; their
archived roots refer to this checkout's local runtime. The frozen declared-input
closure is the portable source and data evidence.

The independent FP64 oracle and represented bundle reader are under `source/`;
the CUDA source is in
[`artifacts/kernels/volta-recurrence/variants`](../../artifacts/kernels/volta-recurrence/variants).
The native bundle, correctness policy, and limitations are summarized in the
[measurement record](../../sources/blogs/volta-recurrence-measurements.md).

Chunking is deferred. This artifact contains no chunk implementation or
equivalence claim; it measures full-sequence retention on already available
inputs and explicit segmented carry.
