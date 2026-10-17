

<!-- todo-orchestrator:v2-managed:start -->
# KWV-04: Build and investigate native GEMM, fragment ownership and software staging

Task revision: `40`; current project revision is in `todo-status.md`.

## Objective
Deliver a correct useful SM70 GEMM baseline and test the actual costs of staging and retaining fragment layouts under a fixed numerical contract.

## State
- Lifecycle: `in_progress`
- Execution: `claimed`
- Parallel policy: `serial`
- Result: `-`

## Next Action
Read KWV-04 in planning/volta-bootstrap/planning/WORK.md, then the listed material. Implement the bounded outcome; reuse existing evidence and tools.

## Ownership
- `exclusive`: `artifacts/kernels/volta-gemm`
- `exclusive`: `benchmark_runs/volta-gemm`
- `exclusive`: `sources/blogs/volta-gemm-measurements.md`
- `exclusive`: `wiki/kernels/volta-gemm.md`
- `exclusive`: `wiki/techniques/volta-fragment-layout.md`
- `exclusive`: `wiki/techniques/volta-software-staging.md`
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
