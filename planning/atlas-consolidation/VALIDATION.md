# Atlas machine and evidence consolidation

Starting revision: `c9bf8dba49f2e5cf90c1dc9a78f970359da5ba5e`.
Source Atlas revision: `5b5629b001da4a42d6f41a84d94f73c07a29d478`.

The consolidation enriches six existing Volta contracts; adds integer/routing
and public VMM contracts, six bounded experiment observations and an E00–E39
status index; and establishes an SM80 route for asynchronous copy, dense/sparse
tensor arithmetic, and resource constraints. Five new source records retain
official-document or benchmark-source classification. No schema replacement,
execution tool, performance measurement, or verification-grade promotion is
introduced. The Atlas and CUDA skill were preserved.

## Executed checks

Run from the KernelWiki root:

```sh
python3 scripts/generate-indices.py
python3 scripts/validate.py
python3 -m unittest discover -s tests -v
python3 scripts/volta_evidence_check.py --all
git diff --check
```

Results: seven indices generated; **1,095 files valid**, 1,010 source IDs,
44 asset bundles, 14 ledgers; **274 tests passed**; **nine archived evidence
cases valid**, zero errors; no whitespace errors. These are CPU/offline checks.
No GPU experiment ran. The architecture tests exercise SM70 versus neighboring
SM72 and SM80/A100, family queries, exact aliases, and confidence filters.

The initial 266-test baseline had four existing failures: three stale
documentation assertions and one positive historical source-closure test
coupled to changed host toolchain files. The repaired tests scope dated audit
receipts correctly, query current corpus state, validate the frozen closure in
the existing portable mode, and independently test that live mode rejects a
same-size toolchain hash change. Production evidence verification and archived
receipt hashes were not changed.

## Retrieval and access

```sh
python3 scripts/query.py ballot --architecture sm70 --type hardware --compact
python3 scripts/query.py --architecture V100 --tag __byte_perm --compact
python3 scripts/get_page.py hw-volta-integer-routing --follow-sources
python3 scripts/get_page.py hw-volta-mma --follow-sources
python3 scripts/query.py --architecture A100 --type hardware --compact
```

The exact integer/routing, MMA, experiment and SM80 routes resolve through the
existing query/get-page interfaces. KernelBender's independent paired check
resolves all 17 contracts in its installed map, their source IDs and actual
architecture/confidence filters. KDA's existing component pins and evaluator
were left unchanged; its CPU consumer/core/engine/selection tests passed 29/29.

The [Atlas source record](../../sources/blogs/atlas-v100-campaign-20261006.md)
identifies private/access-gated origin locators and provides a pinned local
Git fallback. Raw vectors and profiler captures are not in the tracked Atlas
snapshot. New prose is authored synthesis with attribution; Atlas prose and
artifacts were not vendored, and no new license was selected for it.

The [Ampere route](../../references/ampere.md) is scoped to SM80/A100, not full
Ampere-product or CUDA-skill parity. Detailed packed SIMD, topology/NUMA,
platform/non-SM contracts, and most full experiment protocols remain with the
retained source corpora. The [evidence crosswalk](wiki-evidence.md) preserves
the source-only records and unexecuted restricted leads. Further replacement
qualification remains separate work.
