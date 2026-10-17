# KernelWiki — Volta, Hopper & Blackwell Kernel Optimization Knowledge Base
> [!IMPORTANT]
> This skill is maintained as a standalone submodule of
> [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)
> for easy installation.
>
> For bug reports, feature requests, and discussions, please use the main KDA repository:
> https://github.com/mit-han-lab/kernel-design-agents

> **Evidence boundaries:** the audited upstream PR corpus closes at 2026-05-20;
> legacy documentation, releases, and leaderboard snapshots were verified through
> 2026-08-18 UTC. The local Volta studies are dated 2026-10-17; their source and
> execution records carry separate evidence dates.

A structured knowledge base with source-backed Volta/V100 (SM70) hardware guidance and bounded, locally measured native kernel studies, alongside a larger NVIDIA Blackwell (SM100, B200) and Hopper (SM90, H100) kernel corpus. It is packaged as a Claude Code skill. The repository root **is** the skill directory — clone it directly into `~/.claude/skills/` and it works out of the box.

## Install as a Claude Code Skill

```bash
git clone git@github.com:mit-han-lab/KernelWiki.git ~/.claude/skills/KernelWiki
```

That's it. The skill auto-registers (because `SKILL.md` lives at the clone root), and the query scripts auto-resolve the wiki root to their own directory — no environment variable or Python package installation is required. The scripts prefer a host PyYAML installation and transparently use the bundled pure-Python fallback when PyYAML is unavailable (including offline or `python3 -S` environments).

PyYAML remains an optional performance dependency. If a package installation is available, it can be enabled with:

```bash
python3 -m pip install -r ~/.claude/skills/KernelWiki/requirements.txt
```

Smoke test:

```bash
cd ~/.claude/skills/KernelWiki
python3 scripts/query.py --tag nvfp4 --type kernel --compact
python3 scripts/get_page.py kernel-flash-attention-4 --frontmatter-only
```

Optional override for relocating the scripts:

```bash
export BLACKWELL_WIKI_ROOT=/path/to/KernelWiki
```

## What's Here

- Source PR pages, synthesized wiki pages, blog/doc/contest summaries, candidate ledgers, query indices, and artifact bundles.
- Verbatim upstream asset bundles under `artifacts/` (PR patches and complete kernel files or excerpts) — pinned to upstream SHAs via `PROVENANCE.yaml`.
- Auto-generated cross-reference indices — [by architecture](queries/by-architecture.md) / problem / technique / hardware feature / repo / kernel type / language.
- Reviewed candidate ledgers with include/defer/exclude decisions.
- **Hybrid version-claim registry** ([`data/version-claims.yaml`](data/version-claims.yaml)) — per-page `version_sensitive: <id>` pointers + central registry, validated for bidirectional consistency
- **Volta/V100 (SM70)** — six source-reported hardware concept pages sit beside locally compiled, correctness-checked, and measured GEMM, row, attention, grouped/gated, W4A16, and recurrent-state studies. Their results are experimental and limited to the published contracts and shapes; they do not imply global optimality or transfer automatically to SM72, Hopper, or Blackwell.
- [Volta implementation handoff](planning/volta-bootstrap/implementation/IMPLEMENTATION-HANDOFF.md) — compact kernel-family outcomes, exact/family query receipts, source-following and code-access results, and remaining limits.
- Run `python3 scripts/repo_status.py` for current corpus counts.

## Query Tools

All tools run from the skill root, no env var needed.

| Tool | Purpose |
|---|---|
| `scripts/query.py` | Unified search across source and wiki pages (keywords + filters + alias-aware) |
| `scripts/get_page.py` | Fetch any page by `id` or path; `--follow-sources` expands cited sources |
| `scripts/grep_wiki.py` | Regex text search across wiki bodies and PR pages |

Examples:

```bash
python3 scripts/query.py "ping-pong attention" --limit 5
python3 scripts/query.py --tag UMMA --type hardware --compact          # alias → tcgen05
python3 scripts/query.py --architecture B200 --type kernel             # alias → sm100
python3 scripts/query.py --architecture V100 --type hardware --compact # exact sm70
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
python3 scripts/grep_wiki.py "tcgen05" --only wiki
```

## Companion Docs

- [`SKILL.md`](SKILL.md) — Skill entry point: when to engage, 5 navigation paths, output contract.
- [`references/primer.md`](references/primer.md) — Topic map: hardware features, techniques, kernels, symptoms → canonical page IDs.
- [`references/volta.md`](references/volta.md) — Volta/V100 route, with exact-target and family-filter guidance.
- [`references/schema.md`](references/schema.md) — Frontmatter schema, confidence rules, reproducibility ladder, controlled vocabulary, canonical aliases.
- [`references/examples.md`](references/examples.md) — 10 worked query patterns (user question → command sequence → synthesis).
- [`CLAUDE.md`](CLAUDE.md) — Extended schema + navigation reference for Claude Code.
- [`index.md`](index.md) — Human-facing curated top-level index.

## Architecture

Three layers (inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)):

1. **`sources/`** — Raw data. Immutable summaries of PRs, blogs, docs, contests.
2. **`wiki/`** — Synthesized knowledge pages. Cross-referenced by `id`. All have YAML frontmatter.
3. **`queries/`** — Auto-generated cross-reference indices. Do not edit manually; regenerate via `scripts/generate-indices.py`.

Supporting files:
- `data/schemas.yaml` — Required/optional fields per page type
- `data/tags.yaml` — Controlled vocabulary (80+ tags)
- `data/aliases.yaml` — Canonical → synonym mappings
- `data/version-claims.yaml` — Central registry for version-sensitive claims (DEC-1 hybrid)
- `data/tool-versions.yaml` — Snapshot of tracked tool releases (Triton, CUTLASS, CUDA, PTX, …)
- `data/refresh-cutoff.yaml` — Internal refresh-round metadata used by validators
- `candidates/` — Reviewed PR candidate ledgers (per repo)
- `artifacts/` — Verbatim upstream asset bundles, each with `PROVENANCE.yaml`

## Maintenance Tooling

| Script | Purpose |
|---|---|
| `scripts/validate.py` | Validate YAML frontmatter, enforce schema, check link integrity |
| `scripts/generate-indices.py` | Regenerate `queries/*.md` from frontmatter |
| `scripts/generate-pr-pages.py` | Batch-generate source PR pages from candidate ledgers |
| `scripts/repo_status.py` | Print current corpus counts |

```bash
python3 scripts/validate.py
python3 scripts/repo_status.py
python3 scripts/generate-indices.py    # regenerate query indices
```

These commands also work without PyYAML. Installing `requirements.txt` is optional and only selects the host implementation.

## Quality Gates

- `scripts/validate.py` reports 0 validation errors
- `scripts/verify_verbatim.py` verifies upstream-pinned assets
- `scripts/check_dod_fixtures.py` verifies every active Definition-of-Done asset contract and audits retired fixture tombstones
- `scripts/verify_pr_architecture_upstream.py` re-derives a deterministic high-risk PR sample from live paginated GitHub evidence
- `scripts/verify_core_prs.py` verifies generated PR manifests
- `scripts/repo_size_check.py` enforces the repository size budget
- 0 broken links across all internal references
- All `verified` wiki pages have official-doc + upstream-code evidence (enforced by `evidence_basis` field)
- All technique/kernel/language pages have compilable code snippets (`reproducibility >= snippet`)
- Hopper-only wiki pages in the Blackwell-focused corpus explain their `blackwell_relevance`; Volta pages use independent SM70 attribution, and source pages preserve upstream evidence
- Version-sensitive claims (currently Triton 3.6) carry `version_sensitive: <id>` pointers resolving to the central registry

## Scope Rules

- **Architecture scope** — Volta/V100 (SM70) has a separate exact route for both hardware guidance and measured case studies. Blackwell remains the larger kernel corpus; its legacy Hopper-only pages require `blackwell_relevance`, while source pages are exempt. A `volta` family query can include neighboring `sm72` material; exact V100 implementation claims remain `sm70`-scoped.
- **Kernel-only** — No distributed-system topics (DeepEP, DualPipe, EPLB are out of scope).
- **English canonical** — All content in English.
- **First-class DSLs** — CuTe DSL, CUDA C++, PTX, Triton. TileLang / cuTile / JAX-Pallas mentioned but no dedicated guides.

## Repository Layout

```
KernelWiki/                             (= ~/.claude/skills/KernelWiki/)
├── SKILL.md                           # Skill entry point
├── README.md                          # This file
├── CLAUDE.md                          # Extended navigation + schema reference
├── index.md                           # Curated top-level index
├── requirements.txt                   # Optional host PyYAML dependency
│
├── scripts/                           # Query tools + maintenance tooling
│   ├── _yaml_compat.py                # Host PyYAML / offline fallback selector
│   ├── query.py                       # Unified search
│   ├── get_page.py                    # Page fetcher
│   ├── grep_wiki.py                   # Regex search
│   ├── _wiki_root.py                  # Shared root resolver
│   ├── validate.py                    # Schema validator
│   ├── generate-indices.py            # Query-index generator
│   └── generate-pr-pages.py           # Batch PR page generator
│
├── references/                        # Skill knowledge layer
│   ├── primer.md                      # Topic map
│   ├── schema.md                      # Condensed schema reference
│   └── examples.md                    # 10 worked query patterns
│
├── data/                              # Schema + vocabulary
│   ├── schemas.yaml
│   ├── tags.yaml
│   └── aliases.yaml
│
├── candidates/                        # Reviewed PR ledgers (ingestion source of truth)
│   ├── cutlass.yaml
│   ├── sglang.yaml
│   ├── vllm.yaml
│   ├── flashinfer.yaml
│   ├── pytorch.yaml
│   └── deepgemm.yaml
│
├── sources/                           # Layer 1: raw data
│   ├── prs/{repo}/PR-{N}.md
│   ├── contests/{contest}/
│   ├── docs/
│   └── blogs/
│
├── wiki/                              # Layer 2: synthesized knowledge
│   ├── hardware/
│   ├── techniques/
│   ├── kernels/
│   ├── patterns/
│   ├── languages/
│   └── migration/
│
└── queries/                           # Layer 3: auto-generated indices
    ├── by-problem.md
    ├── by-architecture.md
    ├── by-technique.md
    ├── by-hardware-feature.md
    ├── by-repo.md
    ├── by-kernel-type.md
    └── by-language.md
```

## License

Summaries and wiki syntheses in this repository are derivative works citing upstream PRs, blogs, and docs. The tooling (`scripts/`, `references/`, `data/`) is MIT-style; see individual files for any exceptions.
