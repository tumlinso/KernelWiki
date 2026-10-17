# KWV-07 measurement bundle

This bundle contains the exact represented inputs, FP64 semantic references,
contracts, CPU-only identity/precheck records, controller role specs, and
captured native evidence for grouped/gated projections and one defined W4A16
storage path. Source implementations are in
[`artifacts/kernels/volta-grouped-lowbit/variants/`](../../artifacts/kernels/volta-grouped-lowbit/variants/README.md).

The immutable suite bundle is `cases/fixture_bundle/`. Each case binds an
input binary, an FP64 reference binary, and a JSON contract by SHA-256. The
two suite-level `oracles/*-suite.json` files deliberately say they are CPU
prechecks and include no device error values. Native output checks and timing
are recorded separately in `evidence/v1/roles/`.

The grouped suite includes six balanced, skewed, and weighted cases. Assignment
records preserve repeated row/expert choices and scatter through explicit
unique output positions; empty experts remain present. The W4A16 suite includes
an odd K/N packing tail and a shared `K=4096,N=1024` weight matrix at low and
moderate M. Reuse timings execute actual 1/4/16 calls at low M and 1/4 at
moderate M while represented weights/scales/activations remain resident and
unchanged. This is a bounded reuse assumption, not a claim that the weight
matrix is resident in L2 or any other cache.

Run the host tests and fixture identity verifier with:

```sh
python3 -m unittest discover -s benchmark_runs/volta-grouped-lowbit/source/tests -v
python3 benchmark_runs/volta-grouped-lowbit/source/bundle.py verify benchmark_runs/volta-grouped-lowbit/cases/fixture_bundle
make -C artifacts/kernels/volta-grouped-lowbit/variants -f Makefile.cpu cpu-test
```

Run a captured role only through the shared Project Control controller, for
example:

```sh
/home/tumlinson/project-control/scripts/pc-dev python \
  benchmark_runs/volta-gemm/controller/pc_controller.py run \
  --spec benchmark_runs/volta-grouped-lowbit/controller/role-grouped_correctness_baseline.spec.json \
  --json
```

Native timing reports nine-sample CUDA-event medians and synchronized
submit-through-completion host medians separately. The setup timer includes
fixture and FP64-oracle file reads/parsing, validation, allocation, H2D, and
synchronization. Group metadata is measured inside setup and is not additive.
The displayed grouped one-use host total is a derived setup-plus-warm-use sum,
not a measured cold production request. For W4A16, decode and repeated compute
are separate measurements; prep-plus-batch device and setup-plus-prep-plus-
batch host totals are sums of those medians, not one measured cold interval.
Harness setup includes oracle I/O and runs under an initialized CUDA context,
so these models do not predict application startup latency. Reuse keeps each
case's represented weights, scales, and activations unchanged; it does not
claim any L2 or HBM residency.

The interpretation and measured per-case values are in
[`summary.json`](summary.json) and the
[measurement report](../../sources/blogs/volta-grouped-lowbit-measurements.md).
The grouped candidate is faster for the measured balanced/launch-dominated
cases, while the grouped library path is faster on the weighted skewed case.
The materialized W4A16 decode plus dense cuBLAS path has lower resident compute
time than direct decode on all four measured cases, although direct decode
retains the smaller packed representation. These are bounded V100 results, not
a universal dispatch rule. Source closure, compiler, cuBLAS, GPU UUID, exact
shared ELF, and stdout are bound under `evidence/v1/`.
