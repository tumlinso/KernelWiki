# Validation that earns its cost

## 1. Small CPU checks for the shared patch

One focused `tests/test_volta_architecture.py` is sufficient initially. Reuse the project's test conventions; do not introduce a testing framework.

| Check | Expected distinction |
|---|---|
| `sm70`, `sm_70`, `V100`, supported GV100 alias | Same exact target after canonical handling |
| `volta` family | Includes explicit family and recognized family members, with real tags retained |
| `sm70` versus `sm72` | Exact V100 query excludes Xavier-only evidence |
| Generic Volta source | Not automatically assigned exact V100 support without an exact target/product basis |
| Unknown architecture | Remains unknown; no default to Blackwell or Volta |
| Unsupported `sm70a`/`sm700` token | Not accepted as exact `sm70` evidence |
| Known Hopper/Blackwell/product controls | Same filtering and alias behavior as before |
| Query diagnostics | Family queries are not described as exact; no Blackwell-only fallback hint for V100 |
| Two CUDA/PTX language guides | Both appear, deterministically; a new Volta guide does not overwrite the existing guide |
| Generated architecture index | A small synthetic exact/family/unknown fixture matches the filter semantics |

Add tests for a discovered bug, not every imaginable parse combination. Run native schema/link validation and one deterministic regeneration comparison when the content is integrated. No index performance benchmark, fuzzing campaign, giant source-corpus refresh or all-GPU matrix is required.

## 2. Kernel qualification is a different check

Every materialized representative kernel must have an actual SM70 compilation record, an independent correctness comparison, and a clearly described measurement boundary. A Markdown code fence is not compilation. A checksum, finite output, or library call with no checked return status is not adequate correctness evidence. [KW-VALIDATE, AT-LIMITS]

Use exact/FP64 small oracles and matched lower-precision performance baselines as defined in M00. Exercise meaningful dimensions, tails, masks, strides and input changes for each operator. Run memory/synchronization sanitizers on cases that expose those risks, with the existing CUDA controller. Sanity-check shader/collective participation from the supported API contract even if a stress test passes.

## 3. Measuring a real gain

Use the existing benchmark `run_config.json`, `results.json`, `summary.json` and `summary.txt` conventions rather than inventing a second measurement database. Reuse controller/profiler outputs. A thin repository helper may verify links and the required fields; it should not become an experiment scheduler. [SK-BENCH, SK-HOST]

Each candidate comparison records:

- The operator, shapes/distributions/strides, actual represented inputs, numerical policy and independent oracle.
- Source revision, compile flags, binary hash, GPU UUID/capability, actual memory/resource configuration, driver/toolchain and profiler versions.
- Normal unprofiled samples, warmup and repetitions; measurement boundary and excluded preparation costs; baseline and candidate identities.
- Correctness/sanitizer results; selected NCU report/summary and metric availability; the claimed bottleneck and its evidence limits.
- Retained/rejected/inconclusive disposition, failed shapes and the next useful decision.

Use enough repeated useful work to rise above timing granularity, without allowing dead-code elimination or turning a memory-streaming workload into a tiny cache-resident loop unnoticed. Label warm/cache-resident and larger-than-cache cases distinctly. Compare paired/randomized execution order when drift could change the result; observe clocks/power/temperature without changing their policy.

For close results, preserve samples and assess the difference's uncertainty or rerun a short matched comparison. Do not choose a winner from a tiny median ordering inside noise. There is no universal speedup percentage imposed by this plan. The gain must be meaningful for the stated workload and survive the full operator boundary.

Normal timing and replay profiling remain separate. A profiler's stall percentage is not a proof of causality; corroborate with a controlled change and the available metrics. Do not request unsupported Volta counters indefinitely or convert missing values to zero. [EXT-NCU, SK-PROFILE]

## 4. Evidence reuse and promotion

Reuse atlas E09's exact supported coordinate finding; rerun a small compatibility smoke for the new binary rather than repeat all research. Use E06/E10/E16 as limits and existing controls, not fabricated conclusions. The old campaign's `GPU_RUN` status covers its implemented subset only. Do not attach new performance claims to that old campaign or borrow its source hash for new code. [AT-E09, AT-E06, AT-E10, AT-E16]

A useful result can be:

- A correct baseline that exposes the dominant loss.
- A measured winner for a stated shape/reuse regime.
- A rejected variant with a concrete counterexample or performance limit.
- An unresolved causal explanation with a precise measured boundary.

A rejected candidate may close its optimization experiment, but it does not waive the task's required correct representative implementation. An environment failure or unrun test cannot be relabeled a negative performance result.

Keep KernelWiki confidence, reproducibility and optimization maturity distinct. `benchmarked` means measured under a contract, not optimal. Use `source-reported` for independently sourced material unless the native stronger evidence requirements really hold. A new measurement report can use the existing community/benchmark source route; preserve code and data provenance.

## 5. Practical artifact arrangement

Prefer one small shared native support bundle and one source bundle per representative kernel family under `artifacts/kernels/volta-*/full/`. Each captured bundle uses the existing `PROVENANCE.yaml` contract. Compact checked-in result summaries can use the native `bench-record` role. Keep raw NCU captures and measurement vectors in the existing local run directories when repository size limits require it; preserve locators and hashes and state their accessibility.

A proposed thin `scripts/volta_evidence_check.py --case <name>` checks receipt completeness, file/source/binary binding, successful real correctness checks, matching baseline/candidate contracts, and an explicit result disposition. It must reject fabricated/missing results and distinguish `not_run` from `evaluated_not_promoted`. It does not run a new optimizer, enforce arbitrary speed targets or reinterpret the NCU port's unavailable metrics. Bind real host execution gates through the existing supported controller integration when the implementation environment is resolved.

## 6. Release check

Run the compact infrastructure test, native wiki validation, deterministic index regeneration and the representative kernel evidence checks. Inspect a V100 hardware query, a V100 attention query, a broad Volta query and the old Blackwell/Hopper examples. Check source following and code-artifact access. Then provide the validated fork commit for KDA integration. No actual newer GPU is required to demonstrate that unchanged Blackwell/Hopper retrieval still works.
