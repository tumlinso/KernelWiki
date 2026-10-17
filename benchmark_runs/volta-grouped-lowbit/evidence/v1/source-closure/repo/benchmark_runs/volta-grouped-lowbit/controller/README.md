# KWV-07 controller roles

`make_role_specs.py` writes the twelve frozen controller specifications in
this directory. Each suite has correctness baseline/candidate, candidate
Compute Sanitizer, baseline/candidate timing, and one focused candidate NCU
profile. The grouped and W4A16 specs invoke the same `bin/v1/kwv07_runner`.
Only `grouped_correctness_baseline` compiles it; every later role consumes the
same executable bytes.

Run a role from the repository root through the qualified host controller:

```sh
/home/tumlinson/project-control/scripts/pc-dev python \
  benchmark_runs/volta-gemm/controller/pc_controller.py run \
  --spec benchmark_runs/volta-grouped-lowbit/controller/role-grouped_correctness_baseline.spec.json \
  --json
```

For an NCU role, also pass a repository-local shim-log path, for example
`--ncu-shim-log benchmark_runs/volta-grouped-lowbit/evidence/v1/ncu-shim.log`.
The host wrapper forces CUDA 12.9, the matching Compute Sanitizer and Nsight
Compute, and the no-clock/source-import NCU shim. It does not replace host
admission or the shared controller locks.

Both correctness and timing modes execute all declared suite cases. Timing
uses nine positive samples per measured operation and checks every output
again after the last timed invocation. NCU is restricted to the selected
representative case and its profiling-run timing is not benchmark evidence.
