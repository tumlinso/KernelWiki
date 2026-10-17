# Ordered Volta GEMM frontend

This bundle carries the host runner for the bounded SM70 GEMM correctness
checks in `benchmark_runs/volta-gemm-ordered/`. It derives from the runner in
`artifacts/kernels/volta-gemm/variants/`; that earlier measured bundle and its
records remain unchanged.

The host-side correction creates the nonblocking compute stream before input
upload and queues the pageable A, B, and bias copies on that same stream. The
one-hot multi-tile check does the same for A and B. This gives the consumers a
stream-ordering dependency on their uploads without relying on ordering
between the legacy default stream and a nonblocking stream. Device kernels,
case inputs, FP64 references, and numerical tolerances are shared with the
original GEMM investigation.

`build.sh` compiles this runner with the frozen original GEMM source bundle and
pinned CUTLASS v2.11.0 headers. The dense correctness role builds once and
archives the exact ELF under its runner-source SHA. The small-M, ragged, and
one-hot roles reuse that archive. See the benchmark directory for source
closure, role specs, raw controller receipts, and the admitted correctness
results. This frontend has no new timing or profiling claims.
