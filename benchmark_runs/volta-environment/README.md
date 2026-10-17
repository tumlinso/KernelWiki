# Volta environment smoke and receipts

This case qualifies a small CUDA 12.9 native path on the controller-selected V100 SM70, not an optimized kernel family. The bound smoke compiles [`volta_smoke.cu`](../../artifacts/kernels/volta-common/variants/volta_smoke.cu) for `sm_70`. It consumes [`inputs.f16le`](inputs.f16le), two row-major FP16 16×16 matrices, and checks an inline `fma.rn.f32` kernel plus WMMA `m16n16k16` with FP32 accumulation against binary64 host references. WMMA fragment ownership stays opaque; this does not qualify an `mma.sync` lane map.

The [`summary.json`](summary.json) manifest is the machine-checked case record. Run its canonical gate with:

```sh
python3 scripts/volta_evidence_check.py --case environment
```

It binds the source and archived measured ELF, represented input, numerical contract and CPU-oracle record to the baseline/candidate correctness, sanitizer, unprofiled timing and profile controller exports, their foreground leases, and the selected profiler outputs. `binary.path` is the immutable measured ELF archive; `executed_path` identifies the path actually bound by controller receipts. The later native-gate rebuild may replace that executed path without changing the archived bytes.

## Observed results

- Bound native gate: evidence `3209078c-5d10-4a8f-9799-5cc8d725cbc0`; baseline and candidate follow-up correctness runs each returned zero and reported zero maximum absolute error for the PTX FMA and WMMA checks. The same source and binary serve both labels.
- Compute Sanitizer: evidence `3d908fc1-2b99-4ce1-9e98-165f98e9f88d`; memcheck reported zero errors and exited zero, and the application emitted its SM70 `status=passed` record. The controller's raw wrapper summary remains `status=partial`, `conclusive=false`; the case accepts only this narrow zero-error memcheck plus bound application PASS combination. See `controller/sanitizer-root/sanitizer-summary.json` and the unmodified controller evidence export.
- Separate unprofiled CUDA-event timing: baseline and candidate medians are both `2.22720003 us` per launch over seven measured repetitions after one warmup. This is a launch-bound smoke on the same binary, not a speedup comparison or promotion.
- Nsight Compute: evidence `6483d638-6b1c-48fe-af2a-c332c67faf51`; the selected report action is `wmma_16x16x16` on `sm_70`. The report contains 260 metrics; the skill helper found 19 of 127 curated metrics, no source-stall metrics, no rule results, zero PM-sampling series, and unsupported timed-warp samples. The selected action reports `gpu__time_duration.sum=2592.0` and SM throughput `0.005763%` of peak. Collection used controller compact metrics with target-processes `all`; there was no collection-time kernel filter or launch-count limit, so selection happened during post-collection report analysis. Profile duration is not used for performance comparison.

The no-clock policy is explicit: the checked-in NCU shim injects exactly `--clock-control none --import-source yes` for collection. The shim log binds its own path/hash and the actual NCU executable path/hash to controller provenance inputs. No clock or power policy was changed. [`actual-profile.json`](actual-profile.json) records the requested-versus-actual controller adapter scope; it preserves the initial config's proposed full-set/one-launch settings as proposal rather than claiming they ran.

## Environment identity and limits

[`environment.json`](environment.json) records the installed and used tool identities, controller-observed GPU, runtime linkage observations and hashes. The compiler is CUDA `12.9.86`, Compute Sanitizer and Nsight Compute CLI are `2025.2.1.0`, and the controller reports a Tesla V100-SXM2-16GB, `sm_70`, driver `580.178.04`, UUID `GPU-21131915-1488-23af-38dd-1743ae1f5cc8`. The source uses CUDA Runtime APIs. The ELF has no dynamic `libcudart.so` dependency; the installed static runtime archive is identified, but the exact linked archive-member provenance was not extracted. This smoke uses no cuBLAS or CUTLASS library, and the installed-library identity is not a performance claim.

[`results.json`](results.json) summarizes the measured outcomes. [`cpu-oracle.json`](cpu-oracle.json) records the represented-input identity, numerical policy, and the bound smoke comparison outcome. The CPU recomputation helper at [`verify_volta_smoke_oracle.py`](../../artifacts/kernels/volta-common/variants/verify_volta_smoke_oracle.py) independently reproduces the binary64 reference magnitudes and confirms their input hash; the controller-bound native output remains the source of the reported GPU comparison errors. [`make_volta_smoke_input.py`](../../artifacts/kernels/volta-common/variants/make_volta_smoke_input.py) regenerates the exact fixed fixture.

Two failed pre-admission requests are retained under `controller/correctness_baseline/attempts/`; they had no GPU execution. The original failed sanitizer UUID attempt remains in `controller/sanitizer/execution.json`. Neither is counted as a successful run.

CUDA C++ WMMA, CUTLASS C++, CuTe's Python DSL, Triton and explicit PTX MMA mappings are separate paths. This smoke qualifies none of the latter four; see [CUDA C++ scope](../../wiki/languages/volta-cuda-cpp.md) and [PTX scope](../../wiki/languages/volta-ptx.md).
