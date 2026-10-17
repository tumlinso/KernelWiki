

<!-- todo-orchestrator:v2-managed:start -->
# KWV-05: Build row kernels and locate memory/fusion crossovers

Task revision: `44`; current project revision is in `todo-status.md`.

## Objective
Supply stable softmax and one normalization operator; test useful traffic/request and residency tradeoffs without building a reduction library.

## State
- Lifecycle: `in_progress`
- Execution: `claimed`
- Parallel policy: `serial`
- Result: `-`

## Next Action
Read KWV-05 in planning/volta-bootstrap/planning/WORK.md, then the listed material. Implement the bounded outcome; reuse existing evidence and tools.

## Ownership
- `exclusive`: `artifacts/kernels/volta-rows`
- `exclusive`: `benchmark_runs/volta-rows`
- `exclusive`: `sources/blogs/volta-rows-measurements.md`
- `exclusive`: `wiki/kernels/volta-row-kernels.md`
- `exclusive`: `wiki/patterns/volta-memory-and-residency.md`
- `exclusive`: `wiki/techniques/volta-memory-fusion.md`
- `forbidden`: `.gitmodules`
- `read`: `artifacts/kernels/volta-common`
- `read`: `data`
- `read`: `planning/volta-bootstrap`
- `read`: `references`
- `read`: `scripts`
- `read`: `sources`
- `read`: `tests`
- `read`: `wiki`

## Dependencies
- `task`: `KWV-02`
- `task`: `KWV-03`
<!-- todo-orchestrator:v2-managed:end -->
