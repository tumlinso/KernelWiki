# Volta implementation and retrieval handoff

**Snapshot:** 2026-10-17. This note connects the bootstrapped plan's bounded
kernel work to the current KernelWiki pages, runnable source bundles, evidence,
and query routes. It records successful and unsuccessful experiments at their
actual tested scope. It does not claim a globally optimal implementation or a
cross-architecture Volta-to-Hopper/Blackwell port.

## Kernel-family outcomes

| Work | Canonical entry and source | Bounded result | Rejected, inconclusive, or deferred |
|---|---|---|---|
| KWV-04 GEMM | [`kernel-volta-gemm`](../../../wiki/kernels/volta-gemm.md), [`measurement report`](../../../sources/blogs/volta-gemm-measurements.md), [`v1 bundle`](../../../artifacts/kernels/volta-gemm/README.md), [`v1 evidence`](../../../benchmark_runs/volta-gemm/README.md) | Native SM70 comparisons cover dense, small-M, and ragged FP16-input/FP32-accumulation GEMM. cuBLAS led the tested dense/small-M cases. Pinned CUTLASS 2.11.0 measured 34.816 us on the tested ragged case. | PTX was substantially slower on dense/small-M. Ragged PTX-versus-cuBLAS was inconclusive; the CUTLASS point does not establish a general crossover. The separate ordered host frontend is correctness-only; keep any fresh receipts separate from legacy timing. |
| KWV-05 rows | [`kernel-volta-row-kernels`](../../../wiki/kernels/volta-row-kernels.md), [`measurement report`](../../../sources/blogs/volta-rows-measurements.md), [`source bundle`](../../../artifacts/kernels/volta-rows/variants/README.md), [`summary`](../../../benchmark_runs/volta-rows/summary.json) | CTA-row softmax measured 11.264 vs 24.576 us on one 4096-wide row and 47.104 vs 92.160 us on 512×4096. Aligned vector RMSNorm measured 36.864 vs 44.032 us on 512×4096. All are selected V100 cases. | No width crossover, universal vector-width rule, or causal mechanism is established. Fusion was not tested as a complete matched operator. NCU describes one softmax route only. |
| KWV-06 attention | [`kernel-volta-attention`](../../../wiki/kernels/volta-attention.md), [`dataflow technique`](../../../wiki/techniques/volta-attention-dataflow.md), [`measurement report`](../../../sources/blogs/volta-attention-measurements.md), [`source bundle`](../../../artifacts/kernels/volta-attention/variants/README.md), [`summary`](../../../benchmark_runs/volta-attention/summary.json) | Long prefill was 97.280 us materialized vs 96.256 us online (inconclusive). On decode Q=(1,8,1,70), KV=(1,2,4096,70), cuBLAS measured 148.480 us, unsplit SIMT 704.512 us, and corrected split16 including merge 66.560 us. | Split16 is supported only for that exact tested decode route. The supplied-block case checks semantics and correctness only. No planner, 2:4 path, Tensor Core fusion, general split threshold, or measured memory savings. NCU counters are diagnostic; replay timing is invalid. |
| KWV-07 grouped and W4A16 | [`grouped/gated`](../../../wiki/kernels/volta-grouped-gated.md), [`W4A16`](../../../wiki/kernels/volta-w4a16.md), [`grouping/decode technique`](../../../wiki/techniques/volta-grouping-and-decoding.md), [`measurement report`](../../../sources/blogs/volta-grouped-lowbit-measurements.md), [`summary`](../../../benchmark_runs/volta-grouped-lowbit/summary.json) | Paired SIMT won tested tiny/moderate and weighted-balanced groups; split cuBLAS won the weighted-skewed case. For resident W4A16 cases, predecode followed by dense cuBLAS beat the direct software-decoding path, especially with reuse. | Grouped/gated results change with assignment distribution and GEMM engine. The paired route is not a general GEMM engine; no persistent queue was justified. W4A16 is a software storage/decode contract, not a native 4-bit instruction or checkpoint-compatibility claim. |
| KWV-08 recurrence | [`kernel-volta-gated-delta`](../../../wiki/kernels/volta-gated-delta.md), [`measurement report`](../../../sources/blogs/volta-recurrence-measurements.md), [`source bundle`](../../../artifacts/kernels/volta-recurrence/variants/README.md), [`summary`](../../../benchmark_runs/volta-recurrence/summary.json) | Retained state measured 40.768 vs 292.864 us for short prefill and 398.048 vs 6,993.632 us for long prefill, including the per-token launch submission gaps on the baseline. | One-token decode (19.264 vs 19.456 us) is inconclusive. No chunked path was implemented; truncating history or changing the recurrence is outside the contract. |

These comparisons use different operators, fixtures, timing intervals, and
precision contracts. Do not compare their raw latencies across families or
rank them as a single kernel portfolio. Sanitizer wrapper classifications and
profile replay-time caveats are stated on each linked report; raw native output
and tool records remain separately visible. They are not upgraded into claims
of universal sanitizer qualification or general bottleneck explanation.

The ordered GEMM frontend source and controller specifications live in
[`artifacts/kernels/volta-gemm-ordered/variants/`](../../../artifacts/kernels/volta-gemm-ordered/variants/README.md)
with role specifications and receipts under
[`benchmark_runs/volta-gemm-ordered/`](../../../benchmark_runs/volta-gemm-ordered/).
Its [fresh correctness-only results](../../../benchmark_runs/volta-gemm-ordered/RESULTS.md)
pass all six routes on dense, small-M and ragged cases, plus the one-hot check
for both stage counts. All four controller runs bind the same new ELF
`448a2b252a49e1029d12a1180ebf89878318d7ac04ec95257cdde5f769db82ca`.
The frontend orders uploads and computation on one explicit stream. Historical
v1 timings retain their original ELF and do not describe this new build.

## Compact retrieval and source-access checks

The following commands were executed locally on 2026-10-17 with the repository
query tools. Their results are navigation evidence, not additional kernel
benchmarks:

| Query | Result | Interpretation |
|---|---|---|
| `query.py --architecture V100 --type hardware --compact --limit 12` | 6 pages: the six `hw-volta-*` concept entries | Exact-target hardware route remains separate from kernel-study evidence. |
| `query.py --architecture V100 --type kernel --compact --limit 20` | 6 entries: GEMM, rows, attention, grouped/gated, W4A16, recurrent state | All current native V100 families are retrievable under exact SM70. |
| `query.py --architecture volta --type kernel --compact --limit 20` | 6 entries | Broad family route works; interpret it as family-level evidence, not a V100 compatibility guarantee. |
| `query.py --tag attention --type kernel --compact --limit 12` | 7 entries, including `kernel-volta-attention`, FA4, FlashMLA, NSA, and sparse MLA | Attention retrieval surfaces both the new V100 result and unrelated exact-architecture studies. Filter architecture before transferring code or performance. |
| `query.py --architecture H100 --type kernel --compact --limit 10` | 8 entries | Hopper retrieval remains available; results include DeepGEMM, FlashMLA, FP8 block-scale GEMM, MoE, gated-delta, grouped GEMM, and sparse MLA. |
| `query.py --architecture hopper --type kernel --compact --limit 12` | 8 entries | Hopper family alias resolves to the expected kernel set. |
| `query.py --architecture B200 --type kernel --compact --limit 10` | 10 returned entries (limit 10) | Blackwell retrieval remains available; results include FA4, DeepGEMM, FlashMLA, grouped/fused MoE, and NVFP4 GEMM. |
| `query.py --architecture blackwell --type kernel --compact --limit 12` | 12 returned entries (limit 12) | Blackwell family retrieval also works. The different limits prevent a cardinality comparison with the B200 result above. |

Source-following and code-access checks:

- `get_page.py hw-volta-mma --follow-sources` returned the PTX ISA, CUDA tuning
  guide, and atlas source excerpts. These hardware pages remain source-reported
  and do not expose a native code artifact.
- `get_page.py kernel-volta-gemm --follow-sources` returned both the historical
  atlas measurement source and the local GEMM measurement report.
- `get_page.py kernel-volta-attention --follow-sources` returned the local
  attention measurement source. `--include-code` enumerated the local
  `artifacts/kernels/volta-attention/variants/` bundle, including runner and
  online/split implementation source.
- `get_page.py kernel-deepgemm --follow-sources` returned its blog and PR
  sources. Its artifact directory contains captured SM90 and SM100 source
  files with `PROVENANCE.yaml`.
- `get_page.py kernel-flash-attention-4 --follow-sources` returned the paper
  record and blog source. The page has no `artifact_dir`; its snippets and
  references should not be mistaken for a full locally vendored runner.

Generated query indices are part of this navigation surface. Regenerate them
with `python3 scripts/generate-indices.py`; do not edit `queries/*.md`
manually. The exact/family boundary and all measured Volta entries are also
linked from `README.md`, `SKILL.md`, `CLAUDE.md`, `index.md`,
`references/volta.md`, and `references/primer.md`.

## Portable evidence checks

From the installed KernelWiki root, run:

```bash
python3 scripts/volta_evidence_check.py --all
```

This checks nine historical cases across the six representative family
summaries. It uses committed source/input closures, ELF archives, controller
exports and lease copies, without requiring the live Project Control runtime
or an installed CUDA/NCU toolchain. Historical absolute vendor paths remain
captured identities. This checks archived evidence, not fresh GPU execution.
The separate ordered-GEMM correctness continuation is documented in its
`RESULTS.md`; it does not add a performance case to this gate.

The checker passed 88 focused tests. Clean relocated validation passed all
nine cases with historical checkout and vendor file reads blocked; altered
ELF, source, lease and nested-output bindings were rejected. Independent review
also verified the receipt hashes and source-input closure. These checks do not
establish security isolation. Rebuilding native code still requires the pinned
dependencies described by each bundle.

Generated Todo state and live runtime files remain on the execution host and
are ignored by Git. Human-readable task projections remain tracked. The
recurrence collector's full workspace snapshots are local-only; its committed
declared-input closure supplies the source and data needed by this check.

## Remaining limits

- The corpus is a small measured V100 kernel family, not a broad shape sweep,
  production inference stack, or proof of optimality.
- A local correctness pass only qualifies its fixed represented inputs,
  oracle, tolerance, toolchain, binary, and selected cases. It does not prove
  arbitrary-input correctness or transfer a modern kernel to SM70.
- Hardware concept guidance remains `source-reported`; local native case
  studies remain `experimental`. Keep these evidence levels distinct.
- Keep profile-counter observations separate from unprofiled timing. Selected
  NCU captures have partial summaries and replay timing is not used for
  performance claims.
- The `volta` family filter may include `sm72`; only exact `sm70` page
  attribution supports the published V100 measurements.
- Broader dimensions, causal/attention layouts beyond the fixed contract,
  backward kernels, additional quantization formats, persistent queues, and
  recurrent chunking remain deferred unless a concrete use case warrants them.
