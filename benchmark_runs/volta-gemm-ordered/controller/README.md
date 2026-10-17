# Ordered GEMM correctness roles

`role-large_dense-correctness.spec.json` is the sole build role. Its
controller-managed `benchmark.build_argv` invokes the ordered bundle's
`build.sh`, which archives the output ELF at the path that includes the runner
source SHA. The small-M, ragged, and one-hot roles bind and reuse that exact
ELF. The three shape roles call `compare_ordered.py`, a thin adapter around
the existing six-route matrix with its phase fixed to correctness; one-hot
executes the existing multi-tile smoke.

After the ordered source files and this controller directory are integrated,
run `run_correctness_roles.sh` from the repository. It verifies the complete
source closure before the first controller call, uses the established
`pc-dev python` plus repository controller wrapper, and stops on a failed
role. Each result is exported by `export_receipt.py` through read-only
`BackgroundStore(Path(root), create=False).result(evidence_id)`; the exporter
preserves controller JSON, the receipt, local lease, target stdout/stderr, and
the output and archived ELF hashes. It does not use raw SQL or replay a run.

The prepared roles pin V100 UUID
`GPU-21131915-1488-23af-38dd-1743ae1f5cc8` and CUDA 12.9. They do not request
sanitizer, timing, or profiler execution. The CUTLASS v2.11.0 checkout and
CUDA 12.9/cuBLAS paths are external declared inputs and must still match the
source closure when these roles run.
