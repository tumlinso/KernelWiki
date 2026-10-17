# M00 — A usable execution contract, not a new toolkit

## Material to publish

One concise Volta CUDA/CUTLASS guide and one PTX/feature-boundary guide, linked from `references/volta.md`. Keep tool-version statements scoped to the tested compiler/library combination. A source file or fatbin containing `sm70` is not proof that a requested operator actually dispatches to a working native path. Atlas E36 explicitly did not establish that. [SK-OPT, AT-LIMITS]

## Implementation assumptions

The user supplies **`~/ncu-report-skill` with working Volta support**. Inspect its `SKILL.md`, scripts and compatibility report locally. Use its supported commands rather than guessing flags from the Blackwell version. Record its commit and the actual NCU version in evidence. A failed smoke check is an environment dependency to resolve, not authorization to port the profiler inside this task.

Use a CUDA 12.x installation that actually builds the target and supplies a compatible sanitizer. Build only `sm_70` during tuning, with optimization and line information, not `-G`. Record flags; `--use_fast_math` is a separate numerical experiment, not an invisible default. CUDA 13's removal of older native compilation does not imply that a compatible earlier toolkit is unusable. [EXT-CUDA-SUPPORT, SK-OPT]

Prefer standalone native kernels and library baselines. No framework rebuild is required merely to publish a representative kernel. If an existing PyTorch harness is useful, use its actual current stream and verify the extension build and runtime dispatch; do not copy a default-stream example into asynchronous framework code uncritically.

## What to reuse

Use the existing CUDA controller and its structured benchmark conventions. Builds belong in `benchmark.build_argv`, before the GPU lease. Read `cuda/references/execution/host-execution.md`; expand user paths at runtime and select actual GPU UUIDs, not historical ordinals. Compiler and sanitizer must come from the same validated toolkit. No persistent background campaign or power/clock changes. [SK-HOST, SK-BENCH]

For profiling, use the ported NCU skill inside the same host-admission controls. It diagnoses a selected hot kernel; ordinary, unprofiled runs supply timing. Request only the available counters that distinguish the candidate explanations. Unsupported counter data is missing, not zero. Nsight Systems is useful when launch/transfer/setup ambiguity remains; it is not a mandatory separate campaign for every small kernel. [SK-PROFILE, EXT-NCU]

## Software support is not one bit

| Route | Initial treatment |
|---|---|
| CUDA C++ / WMMA / explicit supported PTX | Primary native implementation routes; compile and execute exact forms |
| CUTLASS `Sm70` / CuTe C++ | Reuse compatible implementation/layout sources; pin a revision that builds under the selected toolkit |
| cuBLAS / cuBLASLt | Independent performance baseline; query actual algorithm/epilogue availability for the type and shape |
| CUB / compatible cuSPARSE | Baselines for primitive-shaped work; do not reimplement a generic library for its own sake |
| CuTe Python DSL | Its documented minimum architecture is not the same as CuTe C++; do not infer V100 support from a helper accepting the string `70` |
| Triton | Audit the exact lowering and version. Removal of MMA-v1 support can leave an FMA path; successful compilation is not evidence of Tensor Core use |
| Other DSLs | Use only an already-working, evidenced SM70 route. A compiler port is outside this bootstrap |

Sources: EXT-CUTLASS-MMA, EXT-CUTLASS-LAYOUT, EXT-CUBLAS, EXT-CUTE-DSL, EXT-TRITON. Historical headers are implementation precedents, not a requirement to downgrade every dependency to that exact release.

## Every kernel has two references

**Semantic oracle:** compute the defined operator on the actual represented input values, using FP64 or exact arithmetic on small cases where appropriate. This detects wrong equations, scale placement, masks, state evolution and indexing.

**Matched numerical/performance baseline:** implement the same permitted input, accumulation, intermediate-rounding and output policy. This distinguishes a layout optimization from a precision shortcut. Compare both against the semantic oracle. A baseline with identical bugs is not an independent oracle.

Record input storage type, conversion rule, accumulation type, intermediate storage/rounding, output type, exceptional-value policy, and tolerance before tuning. Tolerance is justified from the operation and downstream use—not enlarged until a candidate passes. FTTN and atlas E10 show why a few random matrices do not characterize accelerator arithmetic. [EXT-FTTN, AT-E10]

For Tensor Core paths, FP32 accumulation does not recover information lost in FP16 input conversion. For fused attention, FP32 running softmax/output state does not imply that the probabilities passed into the next MMA were never rounded to FP16. Make that conversion explicit.

## One bounded environment check

Confirm device identity/capability, native compilation, one correctness test, matching sanitizer, normal timing, and one readable NCU profile. Capture a compact environment record with toolchain and binary/source identities. Reuse it while those identities remain unchanged. Do not turn this into a universal host qualification suite.

Output: a working measurement path and one concise capability table. The subsequent tasks build the actual useful kernels.
