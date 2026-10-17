# Ordered GEMM frontend correctness results

The controller-admitted correctness-only sequence completed on the pinned
Tesla V100 (UUID `GPU-21131915-1488-23af-38dd-1743ae1f5cc8`) with CUDA 12.9.
The source tree checkpoint is `293e6d319db53c45cff3e67ca2406660f2ddd0f7`.
The ordered runner source SHA256 is
`9ab1ba79270f128d5b3cbbd56da8db991799326604ef87973085981672dd3e9c`. The
single controller build produced output and archive ELF files with the same
SHA256:
`448a2b252a49e1029d12a1180ebf89878318d7ac04ec95257cdde5f769db82ca`.

| Role | Controller evidence ID | Routes | Outputs checked per route | Maximum absolute error |
| --- | --- | ---: | ---: | ---: |
| `large_dense` | `8aecb574-2ac7-4615-a81f-9eb1dc3c262e` | 6 | 1,048,576 | 0 |
| `small_m` | `cd7786f8-66a8-40b2-b29e-db88272e2720` | 6 | 65,536 | 0 |
| `ragged` | `2cd11e64-20e9-46b7-9380-cb7564d8cd6c` | 6 | 65,151 | 0 |
| one-hot multi-tile | `2e19b23b-8ba3-42c2-be38-d1e3cc18e5ec` | 2 stages | 16,384 | exact |

The six routes for each shape are cuBLAS, PTX stage 1, PTX stage 2, CUTLASS,
unfused cuBLAS bias/ReLU, and fused PTX bias/ReLU. Every route returned a
passing native `KERNELWIKI-GEMM-CHECK/1` result against the immutable FP64
reference, with zero failed or nonfinite outputs and zero maximum absolute
error. The one-hot check passed exactly for both PTX stage counts. All four
controller records are `succeeded` and `valid`; their local receipt directories
contain the raw controller JSON, read-only evidence export, lease copy, target
stdout/stderr, and output/archive ELF hashes:
`evidence/attempts/kwv09-ordered-check-20261017T02/`.

The build manifest is
`evidence/shared/runner-build-source-9ab1ba79270f128d5b3cbbd56da8db991799326604ef87973085981672dd3e9c.json`
(SHA256 `d35d5b47e93bf167daf5f864fd2c331993dab5347d8794ef9372e4808e3ba8c2`).
The exact archive is
`evidence/shared/bin/gemm_runner-source-9ab1ba79270f128d5b3cbbd56da8db991799326604ef87973085981672dd3e9c.bin`.

These results establish only correctness for the listed cases and routes. No
sanitizer, timing, or profiler role ran, and the historical KWV-04 timing
records do not describe this separately built frontend. Rebuilding requires
the declared CUDA 12.9/cuBLAS installation and the pinned CUTLASS v2.11.0
checkout at the paths in the source-closure manifest; those external inputs
were hash-bound by the captured controller records but are not vendored here.

An initial pre-admission attempt stopped at build because NVCC rejected the
rpath option. That raw failure remains at
`evidence/attempts/kwv09-ordered-check-20261017T01/large_dense/controller.json`.
The corrected build command uses explicit `-Xlinker` arguments; it passed in
the successful build above.
