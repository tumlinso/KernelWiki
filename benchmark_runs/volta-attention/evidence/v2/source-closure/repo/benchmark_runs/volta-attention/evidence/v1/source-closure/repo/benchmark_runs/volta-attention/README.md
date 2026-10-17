# KWV-06 Volta attention measurement bundle

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
and use CUDA 12.9. The first baseline correctness role is the sole native
build. Later roles reuse the byte-identical checked ELF at
`bin/volta_attention_runner`; each controller receipt also captures its own
binary/source/data identities. Timing uses the runner's explicit-stream CUDA
events and excludes allocation, uploads, and correctness readback while
including the full selected implementation. The conditional split-KV specs
are run only if the unsplit decode profile shows a meaningful low-grid limiter.

Run the CPU-only checks with:

```sh
python3 -m unittest discover -s benchmark_runs/volta-attention/source/tests -v
```

For controller roles use the Project Control development launcher and the
qualified `volta-gemm/controller/pc_controller.py` wrapper recorded by each
spec. Do not invoke a runner directly on the shared GPU.
