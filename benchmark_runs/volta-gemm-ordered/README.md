# Ordered Volta GEMM frontend checks

This separate runner variant corrects host upload ordering for the frozen
KWV-04 GEMM implementation. The original measured bundle and timing records
remain under `volta-gemm/`; no timing or profiling result is transferred to
this frontend.

`controller/run_correctness_roles.sh` performs source-closure preflight and
then runs four admitted foreground roles on the pinned V100 UUID. The dense
role compiles once through the controller and archives the exact ELF by runner
source hash. Small-M, ragged, and one-hot roles reuse that archive. The first
three cases invoke all six existing GEMM routes against the immutable
independent FP64 references: cuBLAS, PTX stage 1 and 2, CUTLASS, and the two
bias/ReLU epilogue routes. The final role checks both PTX stage counts with the
exact one-hot multi-tile output.

Each attempt preserves controller JSON, the read-only evidence export, the
foreground lease, target stdout/stderr, and explicit output/archive ELF
hashes. These roles establish bounded correctness for the checked contracts;
they do not run a sanitizer, timing, or profiler campaign.
