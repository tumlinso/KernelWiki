

<!-- todo-orchestrator:v2-managed:start -->
# KWV-03: Qualify the existing native measurement path and reusable receipts

Task revision: `13`; current project revision is in `todo-status.md`.

## Objective
Use the CUDA controller and user-supplied Volta NCU skill to establish a small working compilation/correctness/timing/profile path. Do not build another toolkit.

## State
- Lifecycle: `in_progress`
- Execution: `claimed`
- Parallel policy: `serial`
- Result: `-`

## Next Action
Read KWV-03 in planning/volta-bootstrap/planning/WORK.md, then the listed material. Implement the bounded outcome; reuse existing evidence and tools.

## Ownership
- `exclusive`: `artifacts/kernels/volta-common`
- `exclusive`: `benchmark_runs/volta-environment`
- `exclusive`: `scripts/volta_evidence_check.py`
- `exclusive`: `tests/test_volta_evidence_check.py`
- `exclusive`: `wiki/languages/volta-cuda-cpp.md`
- `exclusive`: `wiki/languages/volta-ptx.md`
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
- `task`: `KWV-01`
<!-- todo-orchestrator:v2-managed:end -->
