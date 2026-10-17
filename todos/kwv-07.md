

<!-- todo-orchestrator:v2-managed:start -->
# KWV-07: Implement grouped/gated math and one exact low-bit storage path

Task revision: `55`; current project revision is in `todo-status.md`.

## Objective
Reuse the native matrix substrate to test grouped/gated fusion and one W4A16 decoding contract; avoid a MoE runtime or format-compatibility project.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `serial`
- Result: `validated`

## Next Action
Read KWV-07 in planning/volta-bootstrap/planning/WORK.md, then the listed material. Implement the bounded outcome; reuse existing evidence and tools.

## Ownership
- `exclusive`: `artifacts/kernels/volta-grouped-lowbit`
- `exclusive`: `benchmark_runs/volta-grouped-lowbit`
- `exclusive`: `sources/blogs/volta-grouped-lowbit-measurements.md`
- `exclusive`: `wiki/kernels/volta-grouped-gated.md`
- `exclusive`: `wiki/kernels/volta-w4a16.md`
- `exclusive`: `wiki/techniques/volta-grouping-and-decoding.md`
- `forbidden`: `.gitmodules`
- `read`: `artifacts/kernels/volta-common`
- `read`: `artifacts/kernels/volta-gemm`
- `read`: `data`
- `read`: `planning/volta-bootstrap`
- `read`: `references`
- `read`: `scripts`
- `read`: `sources`
- `read`: `tests`
- `read`: `wiki`

## Dependencies
- `task`: `KWV-04`
<!-- todo-orchestrator:v2-managed:end -->
