# Native plan intake

## Install the package, not a replacement authority

Copy this directory into the standalone KernelWiki checkout as:

`planning/volta-bootstrap/`

The plan's repository-relative references and gate paths assume that location. Do not copy the `seed/` subtree onto canonical wiki paths blindly; source/material adoption is an explicit task after architecture registration.

The package contains native **Todo schema v3**, not an alternative task format. Its fields follow the inspected `src/todo_orchestrator/plan.py`, gate definitions and working v3 plan examples in the Project Control workspace. Its ten tasks comprise nine concrete outcomes and one aggregate completion task.

## Validate and review before applying

Use the installed Project Control environment and CLI, whose tests exercise this grammar:

```bash
PLAN="$(pwd)/planning/volta-bootstrap/planning/kernelwiki.todo-plan.json"
project-control plan validate --project kernelwiki --file "$PLAN"
```

The native validation result includes prospective additions/modifications. Review that result, project identity, active claims, and existing tasks. The IDs in this package are namespaced `KWV-`; do not assume they are unused on a later checkout. Inspect `project-control plan --help` if the installed release differs. Do not run a globally unrelated Python installation or install another Todo package to make the command work.

After native validation succeeds and the prospective change contains only the intended plan:

```bash
project-control plan apply --project kernelwiki --file "$PLAN"
```

These commands were **not run on the user's host during package preparation**. Current source evidence for the grammar is `project-control/tests/test_cli_profiles.py` (the plan tests) and `tests/test_plan_apply.py`. The native authority remains responsible for freshness checks and transactional application. The package's own checker does not substitute for that validation.

The previous observer error `todo_read_database_open_failed` means authority access was unavailable; it does not mean no project exists. Resolve that through the normal owner-maintained environment. Do not initialize a new project, change UUIDs, clear claims, migrate state, or overwrite a database automatically.

## Execution sequence

The plan uses a single exclusive implementation lane and the normal Project Control task lifecycle. It has no coordinator-only task and no parent-as-prerequisite cycle. The aggregate task closes only after its children are complete. Kernel tasks may share existing bounded helper/delegation mechanisms; there is no new concurrency framework to implement.

Some command gates point to the small test/evidence helpers that the plan explicitly asks the implementing agent to create. That is intentional: the package supplies their behavioral contracts, not untested implementation code. Before the first hardware run, bind actual toolchain/controller execution gates using the installed supported interface. Do not guess NCU port flags or bypass CUDA admission.

## Validation status at delivery

- JSON, task graph, internal source/material links and seed metadata: checked locally as recorded in `validation/package-check.json`.
- Native Todo validation/application: pending on the user's installed authority.
- Native KernelWiki validation after changes: pending implementation.
- NVCC compilation, sanitizer checks, GPU correctness, profiling and speedups: not performed in this package.
