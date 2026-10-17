# KWV-06 Volta attention measurement bundle

The completed results and applicability limits are in the
[measurement report](../../sources/blogs/volta-attention-measurements.md).
The evidence checker validates all eleven captured roles, including the
corrected 16-way split on `decode70-causal`:

```sh
python3 scripts/volta_evidence_check.py --case attention
```

This directory holds the independent CPU oracle and deterministic represented
inputs, native-runner specifications, and task-scoped execution evidence for
forward attention on the admitted SM70 host. Source implementations are in
`artifacts/kernels/volta-attention/variants/`.

`cases/attention/attention_inputs.bin` stores FP16-represented Q/K/V for the
fixed seven-case suite. `attention_reference.bin` stores FP64 semantic outputs
and row states. The adjacent contract, case index, per-case metadata, and source
manifest bind those exact bytes and definitions. The standard contract domain
is bounded to finite `abs(Q,K,V) <= 0.25`, `D <= 128`, and `Sk <= 4096`; the
`score-stress64-noncausal` fixture has a separate declaration at `abs(Q,K)=4`
and does not broaden that domain. Fixed pre-run tolerances are in the contract.

Controller specs target only UUID `GPU-21131915-1488-23af-38dd-1743ae1f5cc8`
and use CUDA 12.9. The captured v1 baseline correctness role built
`bin/volta_attention_runner` once; the other v1 roles reused that exact ELF.
After the merge ownership correction, the v2 split correctness role built
`bin/v2/volta_attention_runner` once, and its sanitizer/timing roles reused
that second ELF. Both executables and their complete source input snapshots
are archived under `evidence/`. Each controller receipt captures its own
binary/source/data identities. Timing uses the runner's explicit-stream CUDA
events and excludes allocation, uploads, and correctness readback while
including the full selected implementation. The conditional split-KV specs
were executed only after the unsplit decode profile showed a low-grid limiter.
The v1 specs and source manifest describe historical captures. For a fresh
corrected split run, use the explicit `--splits 16` specs in `controller/v2/`;
the correctness spec contains its build command. Keep new run records separate
from these frozen captures.

Run the CPU-only checks with:

```sh
python3 -m unittest discover -s benchmark_runs/volta-attention/source/tests -v
```

For controller roles use the Project Control development launcher and the
qualified `volta-gemm/controller/pc_controller.py` wrapper recorded by each
spec. Do not invoke a runner directly on the shared GPU.
