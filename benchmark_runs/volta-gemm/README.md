# KWV-04 Volta GEMM records

This directory preserves the contracts, exact represented inputs, independent
CPU references, native runner, controller records, and analyses for the SM70
GEMM comparison. Start with the [kernel result](../../wiki/kernels/volta-gemm.md),
the [measurement report](../../sources/blogs/volta-gemm-measurements.md), or the
[source bundle](../../artifacts/kernels/volta-gemm/README.md).

## Results covered

The primary matched comparison is cuBLAS against the two-stage PTX `mma.sync`
candidate. Each pair used the same represented inputs and V100 UUID, with
11 unprofiled CUDA-event samples per implementation.

| Case | Shape `(M,N,K)` | cuBLAS median (us) | PTX stage 2 median (us) | Recorded interpretation |
|---|---:|---:|---:|---|
| `large_dense` | `(1024,1024,1024)` | 61.440 | 232.448 | Retain cuBLAS for this dense case. |
| `small_m` | `(16,4096,4096)` | 83.968 | 244.736 | Retain cuBLAS for this small-M case. |
| `ragged` | `(127,513,769)` | 50.176 | 47.104 | PTX/cuBLAS result is inconclusive after the supplemental paired matrix. |

The supplemental matrix measures cuBLAS core, PTX stages 1 and 2, CUTLASS
2.11.0, cuBLAS plus a separate bias/ReLU kernel, and fused PTX bias/ReLU.
For the tested ragged contract, CUTLASS measured 34.816 us against the
supplemental cuBLAS median of 49.152 us. This is a shape- and configuration-
scoped result, not a general dispatch rule. The full sample vectors, paired
differences, correctness details, evidence IDs, and profiler qualifications
are in the measurement report and `supplemental/` records.

## Evidence map and checks

- `cases/` contains the immutable workload contracts, represented inputs,
  oracle metadata, and FP64 references.
- `evidence/shared/bin/` preserves the exact measured runner binary. `bin/`
  is the original local execution/build location and is not part of this archive.
- `controller/` contains the six primary controller role specifications per
  case and the short one-hot smoke specification.
- `evidence/<case>/<role>/` contains primary controller execution records,
  exported evidence, output, and lease copies. Selected Nsight Compute data
  lives under the profile-role directories.
- `supplemental/` contains the additional variant correctness/timing
  collections and their controller records.
- `summary.json` binds the three primary cases. The corresponding original
  foreground lease paths and SHA-256 values are enumerated in
  [`LEASE-MANIFEST.json`](LEASE-MANIFEST.json); archived copies are recorded
  beside each role. The current evidence checker also reads the original
  `.todo-orchestrator/runtime/background-artifacts/foreground/` paths, so
  preserve those 18 runtime lease files with this checkpoint. Porting those
  bindings to an archive-only representation is outside this KWV-04 record.

From the repository root, the focused native checks are:

```sh
python3 scripts/volta_evidence_check.py --case gemm
python3 scripts/validate.py
```

The saved controller specifications document the original CUDA 12.9 host and
include an absolute temporary CUTLASS checkout path. They preserve how the
recorded runs were made but are not yet portable rerun recipes. CUTLASS is
pinned to v2.11.0 commit `66d9cddc832c1cdc2b30a8755274f7f74640cfe6`; its
upstream documentation describes CUDA 11.x, while this record establishes
only the observed local CUDA 12.9 build and runs. The raw memcheck output is
zero-error and the native output checks pass, but the sanitizer wrapper
classification remains `partial`/`conclusive=false`. Selected profiler
captures are single-kernel samples with partial summaries and do not establish
a performance cause.
