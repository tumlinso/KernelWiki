# Volta bootstrap readiness

Prepared and bootstrapped 17 October 2026. **Implementation is authorized and
underway**, using Luna subagents with root integration and acceptance.

`KWV-01` and `KWV-02` have completed their native gates. `KWV-03` is active:
the controller-admitted SM70 correctness smoke passed, and a matching-toolkit
memcheck completed with zero reported errors. Timing/profile qualification is
still pending. These observations qualify the smoke only, not later kernels.

The historical PR receipts and all 944 PR pages remain unchanged. Their exact
pre-Volta policy is preserved as non-executable provenance in
`audit/policies/`, with a pinned origin manifest and negative validation tests.
This does not claim full reclassification under the new policy or a new live
upstream audit.

## Bootstrap receipt

Fresh native validation passed, then `project-control plan apply` created
`KWV-000` and `KWV-01` through `KWV-09`, with no existing tasks modified and
no warnings. Authority revision advanced from 1 to 2. Run: `KWV-RUN-1`.
The supported application regenerated Todo projections and the portable snapshot.
No implementation task was claimed or started. Application required the approved
sandbox escalation because the existing SQLite authority is in protected Git
metadata; no replacement database was created.

## Installed input

The complete 45-file attachment is extracted at `planning/volta-bootstrap/`,
the location expected by its plan references and gate paths. Delivered files
are unchanged. The original attachment remains in place. This readiness note
is outside the package so its manifest stays valid.

Baseline HEAD: `d5d2637aa9651e5fad835ea5192c20a2393da8ab`.
The working tree was clean before extraction. All eleven KernelWiki input
hashes in `sources/local-inputs.json` match current files.

## Suggested implementation

Extend the existing architecture policy, vocabulary, query diagnostics and
navigation for Volta/V100/SM70. Keep SM72 attribution distinct. Support multiple
language guides in the native index. Adopt the 11 draft source pages, six
draft hardware pages and routing reference only after architecture registration.
Then investigate GEMM/layout, row operations/fusion, attention, grouped/gated
and low-bit operations, and recurrent state/chunks. Deeper work is conditional.
Preserve the existing Hopper/Blackwell corpus and independent source projects.

The schema-v3 plan proposes nine implementation outcomes plus aggregate
`KWV-000`, one exclusive implementation lane and 21 gates. Aggregate task-state
gates require children to become done; they do not mark them already complete.

## Checks performed here

- Package checker with `--verify-manifest`: passed, including seed metadata;
  no warnings. All delivered files match the manifest.
- Existing corpus baseline `python3 scripts/validate.py`: passed; 1,041 files,
  37 asset bundles and 14 candidate ledgers. This does not validate an adopted
  Volta extension, which has not been implemented.
- Native `project-control plan validate`: passed; ten additions (`KWV-000`,
  `KWV-01` through `KWV-09`), no modifications and no warnings, at authority
  revision 1. Project UUID: `c6ca94a4-a877-417d-876d-a8dc94221905`.
  Plan digest: `51cdc61ebe07397f2f6852fce77f82c93f6ac1ca22e3d6bbaac4a2e589a86a2b`.
- Project Control overview: semantic workflow available. The authoring-time
  authority-open error did not recur.
- Optional CPU equation checks: all six groups passed with Python 3.13.11 /
  NumPy 2.2.6 from `/home/tumlinson/Software/miniconda3/bin/python`.
  Default Python and Project Control's Python lack NumPy; no installation is
  needed when using this existing interpreter for that optional check.
- `/home/tumlinson/ncu-report-skill` exists and contains the Volta route and
  recorded source-attribution fixture evidence. That historical fixture is
  not qualification of future KernelWiki kernels.
- Executable version queries: CUDA compiler 12.9.86; Nsight Compute 2025.2.1.0.
  Current paths are `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc`
  and `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/ncu`.

## Continue implementation

The plan is already applied. Do not reapply it merely to begin implementation.
Use the normal Project Control task lifecycle for this run; respect its active
claim before calling `next_task`. The original intake validation command is:

```sh
project-control plan validate --project kernelwiki --file /home/tumlinson/KernelWiki/planning/volta-bootstrap/planning/kernelwiki.todo-plan.json
```

Continue the task lifecycle described in
`volta-bootstrap/planning/INTAKE.md`. Inspect the actual
CUDA controller and profiler interfaces before admitting any GPU work. Bind
compiler, numerical and hardware evidence to actual source and binary identities.

During the initial preparation, no implementation or GPU work occurred. The
later user signal authorized the work recorded above. No clock/power changes
or persistent campaigns are part of this run. The user's subsequent request
also authorizes intermittent commits/pushes and updating their KDA fork's
KernelWiki submodule pointer to published component checkpoints.
