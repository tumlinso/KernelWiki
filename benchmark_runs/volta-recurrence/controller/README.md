# KWV-08 controller roles

`make_role_specs.py` writes exactly six frozen specifications under this
directory. All declare the same inputs and exact executable path. Only
`recurrence_correctness_baseline` owns `benchmark.build_argv`; it builds the
SM70 runner once. Every later role runs that same ELF and the same immutable
input file.

From the repository root, run roles through the qualified controller entry:

```sh
/home/tumlinson/project-control/scripts/pc-dev python \
  benchmark_runs/volta-gemm/controller/pc_controller.py run \
  --spec benchmark_runs/volta-recurrence/controller/role-recurrence_correctness_baseline.spec.json \
  --json
```

Capture the exact controller JSON stdout for each role, plus the shell stdout
and stderr. The profile role also requires the repository-local shim log:

```sh
/home/tumlinson/project-control/scripts/pc-dev python \
  benchmark_runs/volta-gemm/controller/pc_controller.py run \
  --spec benchmark_runs/volta-recurrence/controller/role-recurrence_profile_candidate.spec.json \
  --json \
  --ncu-shim-log benchmark_runs/volta-recurrence/evidence/v1/ncu-shim.jsonl
```

The wrapper selects CUDA 12.9, matching Compute Sanitizer and Nsight Compute,
and the existing NCU collection shim. The shim adds or verifies
`--clock-control none` and `--import-source yes`; its exact received/executed
arguments and executable hashes are preserved. The profile mode selects only
`long_prefill_many_sequences` and launches one full retained-state kernel.
The runner's pre-profile correctness result is useful for interpreting the
report; NCU replay timings are not benchmark evidence.

The fixed admission request is GPU UUID
`GPU-21131915-1488-23af-38dd-1743ae1f5cc8`, two CPU threads, no topology
isolation, and quiescence of three idle samples at 0.25-second intervals with
a 60-second timeout. These settings are identical for all roles.

After each controller result, use `source/collect_native_evidence.py` to
export its read-only `BackgroundStore(..., create=False).result(ID)` record,
lease receipt, foreground stdout/stderr, exact CLI JSON, source snapshot, and
the byte-identical archived ELF. Freeze repository input bytes first with
`source/pack_source_closure.py`. Neither script performs controller runs,
compilation, or GPU work by itself.
