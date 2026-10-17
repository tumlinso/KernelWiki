---
id: pattern-volta-memory-and-residency
title: Reading Memory and Residency Evidence for Volta Row Kernels
type: pattern
architectures: [sm70]
tags: [vectorized-loads, kernel-fusion, shared-memory-optimization]
symptoms: [memory-bound-profile, low-sm-throughput, working-set-exceeds-l2]
candidate_techniques: [technique-volta-memory-fusion]
related: [kernel-volta-row-kernels]
sources: [blog-volta-rows-measurements]
---

# What the row-kernel measurements establish

The measured row comparisons ran on a Tesla V100-SXM2-16GB with 6,291,456
bytes of reported L2. The two-buffer rotated working sets were 65,536 bytes
for the one-row softmax, 33,554,432 bytes for the streaming softmax, and
33,587,200 bytes for streaming RMSNorm. The latter two exceed the L2 capacity;
this labels a capacity relationship, not cache residency or HBM traffic.

Unprofiled 21-sample medians favored CTA softmax for the tested single row
(`11.264` versus `24.576 µs`) and streaming rows (`47.104` versus `92.160
µs`). For the one streaming RMSNorm shape, aligned `float4` measured `36.864`
versus `44.032 µs` for scalar CTA. These measurements establish route
differences for their exact source, shapes, buffers, and inputs. Timing alone
does not establish whether launch overhead, instruction count, memory
transactions, cache behavior, or another mechanism caused a difference.

# How to read the profile

Nsight Compute selected one candidate `softmax_cta(...)` launch from the
streaming softmax case. Its single sample was classified as memory-bound and
reported 44.01% DRAM throughput, 20.30% SM throughput, and 32 registers per
thread; occupancy was not available. The controller summary is partial, marks
counters valid, marks replay timing invalid, and requests more data. Treat this
as a narrow counter observation. It is not timing evidence and says nothing
about the RMSNorm kernel.

# Follow-up checks

For a suspected bandwidth limit, bind counters to the exact hot kernel and
measure actual bytes/transactions alongside the workload's access pattern.
Separate profile replay from ordinary unprofiled timing. For a suspected
residency effect, vary a controlled working set across L2 capacity while
keeping launch count and route stable; capacity alone is not a cache-miss
measurement. For a vectorization change, verify post-offset pointer and
row-stride alignment, tail handling, and scalar fallback before interpreting
performance.

Do not infer shared-memory bank conflicts, occupancy, or HBM traffic from
these row-kernel timings. The selected profile reports no usable occupancy
value, and it does not include RMSNorm. Raw samples, source and binary hashes,
native output checks, role receipts, and the exact profiler capture are bound
in `benchmark_runs/volta-rows/summary.json` and the files under
`benchmark_runs/volta-rows/evidence/v2/`.
