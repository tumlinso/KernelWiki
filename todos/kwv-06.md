

<!-- todo-orchestrator:v2-managed:start -->
# KWV-06: Implement Volta attention dataflow for prefill and decode

Task revision: `48`; current project revision is in `todo-status.md`.

## Objective
Reuse matrix and row work to produce correct forward attention, then investigate fragment exchange, useful query grouping and bounded KV partitioning.

## State
- Lifecycle: `in_progress`
- Execution: `claimed`
- Parallel policy: `serial`
- Result: `-`

## Next Action
Read KWV-06 in planning/volta-bootstrap/planning/WORK.md, then the listed material. Implement the bounded outcome; reuse existing evidence and tools.

## Ownership
- `exclusive`: `artifacts/kernels/volta-attention`
- `exclusive`: `benchmark_runs/volta-attention`
- `exclusive`: `sources/blogs/volta-attention-measurements.md`
- `exclusive`: `wiki/kernels/volta-attention.md`
- `exclusive`: `wiki/techniques/volta-attention-dataflow.md`
- `forbidden`: `.gitmodules`
- `read`: `artifacts/kernels/volta-common`
- `read`: `artifacts/kernels/volta-gemm`
- `read`: `artifacts/kernels/volta-rows`
- `read`: `data`
- `read`: `planning/volta-bootstrap`
- `read`: `references`
- `read`: `scripts`
- `read`: `sources`
- `read`: `tests`
- `read`: `wiki`

## Dependencies
- `task`: `KWV-04`
- `task`: `KWV-05`
<!-- todo-orchestrator:v2-managed:end -->
