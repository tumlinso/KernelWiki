# Decision: extend the native wiki, not its architecture

## Chosen approach

Add Volta entries to KernelWiki's existing source and page-type directories. Extend the existing architecture policy and a few navigation surfaces in place. Do not build an overlay loader, alternate index, new backend framework, or separate Volta wiki. The actual code already discovers `sources/**/*.md` and `wiki/**/*.md` and centralizes architecture evidence in `scripts/pr_policy.py`. This makes a small native patch the lowest-maintenance option. [KW-QUERY, KW-POLICY, KW-INDEX]

This decision supersedes the earlier suggestion to make every legacy policy architecture-neutral immediately. **Volta needs independent applicability, not a wholesale rewrite of Blackwell's editorial rules.** Keep `blackwell_relevance` on existing pages and retain its legacy Hopper-only check. Do not add `hopper_relevance`, `cross_arch_relevance`, or false Hopper/Blackwell architecture tags. Use existing `related` links and explicit prose for comparisons. For this first extension, put cross-generation porting guidance in `references/volta.md`; do not create new migration-page requirements. [KW-VALIDATE, KW-SCHEMA]

## Minimal shared changes

| Surface | Change | Keep unchanged |
|---|---|---|
| `data/tags.yaml` | Add `volta`, `sm70`; recognize `sm72` only as a neighboring Volta-family attribution boundary. Add the few topical tags actually used by adopted pages. | Existing vocabulary, spelling, confidence and reproducibility levels |
| `data/aliases.yaml` | Add `V100`/`GV100`/`sm_70` aliases to `sm70`; keep `volta` a family, not an alias for only one product. | Existing Blackwell/Hopper aliases |
| `scripts/pr_policy.py` | Extend family/exact-target sets, exact-SM and numeric guard parsing, and authoritative product mappings for V100. Recognize `sm72` without assigning its features to `sm70`. | Pure policy functions, provenance rules, scope exclusions, exact/family/unknown semantics |
| `scripts/query.py` | Derive family detection for exclusion messages from the shared policy; replace the hard-coded Blackwell-only hint. Update help text where necessary. | Keyword scoring, filters, result formats, path-only mode, alias system |
| `scripts/generate-indices.py` | Keep architecture generation. Change language-guide storage from a single overwritable value to a deterministic collection; preserve the existing single-guide output. | Native index filenames and general table format |
| `SKILL.md` | Admit V100/Volta questions and route explicitly through the Volta reference with an architecture filter. | Existing query commands, evidence contract and Blackwell/Hopper examples |
| `README.md`, `CLAUDE.md`, `index.md`, `references/primer.md` | Small scope/entrypoint additions, links to Volta material, honest validation wording. | Original corpus, original source cutoff and historical evidence |
| `queries/*.md` | Regenerate after canonical content changes. | No manual editing or parallel generated index |
| `data/version-claims.yaml` | Add a pointer only for a specific, externally supported tool-dependent statement that needs it. | Existing claims, release snapshots, and tool vocabulary |

`sm72` here is **taxonomy-only**, not a request to develop Xavier kernels. This small distinction prevents the words “Volta” or “integer WMMA” from becoming an unsupported V100 capability claim. Exact V100 queries must not return `sm72`-only entries. Family queries may expose them with their real tags. Do not introduce `sm70a`/`sm70f`. [EXT-LEGACY, AT-R04, EXT-PTX]

A source reference to a lower architecture in a compatibility fallback does not necessarily identify the target of an entire PR. Extend existing evidence extraction conservatively; do not turn all `>=700` text or every `Sm70` helper used by newer kernels into blanket runtime support. Preserve the source's exact, family, mixed, or unknown attribution.

### Additional issue discovered in the code

The current query diagnostic has a hard-coded family list and directs users toward Blackwell for broader evidence. Its architecture filter itself is already shared. Fix the diagnostic, not the search engine. The language index's single `guide` slot would silently replace an existing CUDA/PTX guide when a Volta guide is added. A small multi-guide collection is necessary native integration, not an optional indexing redesign. [KW-QUERY, KW-INDEX]

Existing unscoped diagnostic pages may recommend CLC, TMEM, or modern barriers. **Do not bulk-tag these pages `sm70`.** Add a small explicitly titled Volta diagnostic entry or an architecture-scoped route. Following a `related` link supplies background, not automatic compatibility. Existing pattern indexes can remain structurally unchanged when distinct Volta titles and metadata make the boundary clear.

## Native page and evidence contracts

Use filenames such as `wiki/hardware/volta-mma.md`, `wiki/techniques/volta-fragment-layout.md`, and `wiki/kernels/volta-attention.md`, with unique `hw-`, `technique-`, or `kernel-` IDs. Retain the existing metadata structure. The prepared drafts in `seed/` are intentionally hardware/concept pages and source records; kernel pages are authored only with their real implementation artifacts. [KW-SCHEMA]

The native snippet validator recognizes code fences and code-like text; it does not invoke NVCC. Native source/schema validation, actual compilation, executed correctness, and measured performance are four separate checks. Do not describe one as another. `confidence: verified` is an evidence classification, not an optimality certificate. [KW-VALIDATE]

Raw third-party implementation files belong in a properly licensed `artifacts/` bundle with `PROVENANCE.yaml`, not pasted into wiki prose. Use native `artifact_dir` and existing per-file provenance fields. Original or adapted implementations are `derived`, not `verbatim`. Keep pinned upstream identities and local measurements separate. [KW-SCHEMA]

The native `source-doc` type permits official documentation or papers. Do not label an arbitrary GitHub file a paper, or invent a PR to satisfy `upstream-code`. A source-backed engineering report may use the existing blog/community-note route with code/provenance links. Keep its consumer page `source-reported` unless the actual native requirements for a stronger classification are met. Do not add a source type merely to obtain a higher confidence label.

## Why no other architecture framework

| Alternative | Decision |
|---|---|
| Separate Volta content-pack loader | Rejected: the current recursive discovery already works; a second registry increases maintenance. |
| Rewrite architecture policy into a universal capability database | Rejected for this task: a few explicit targets and mappings suffice. |
| Remove all Blackwell-first fields and migrate the corpus | Deferred: not required for valid Volta-only pages and likely to conflict repeatedly. |
| Copy/retag Blackwell kernel pages | Rejected: often changes instructions, arithmetic, or actual applicability without evidence. |
| Reimplement query ranking or benchmark the indexer extensively | Rejected: no demonstrated bottleneck. Use focused regression cases. |
| Port NCU/optimizer first inside this plan | Rejected: user supplies the NCU port. Inspect and consume it, do not duplicate it. |

## Upstream continuity

Before editing, record current branch, HEAD, dirty paths, upstream remote and its actual default branch. Preserve unrelated changes and existing task claims. The source hashes in this package are read snapshots, not permission to overwrite a newer file.

Keep shared patches in small commits separate from Volta source/content additions. Fetch upstream normally; do not assume `master` or `main`. Merge the actual upstream default branch into the maintained fork, resolve canonical code/data conflicts, and regenerate derived indexes. Keep the legacy `BLACKWELL_WIKI_ROOT` override working; renaming it buys no kernel performance.

Run the short architecture/index tests, native validator and targeted changed-content checks. Do not run every live GitHub corpus audit on each edit. Changes to generated PR evidence must use the existing PR machinery and its genuine receipts; source-only Volta additions do not require a wholesale corpus refresh.

The refresh script's bounded keyword search is not an exhaustive historical search. Pin relevant historical CUDA/CUTLASS implementations directly; curate a few genuine PRs only when they add useful evidence. Do not rewrite the global refresh cutoff just because Volta material has a newer retrieval date. [KW-REFRESH]

After validated implementation, publish the fork commit and provide KDA with that exact commit. Updating `kda/skills/KernelWiki` is an integration handoff, not a reason to develop the component inside its submodule. The bootstrap itself does not change KDA or another workspace.

## Minimal acceptance for this decision

V100 and family filters work; older exact targets do not leak into one another; two language guides coexist; legacy examples still work; new pages are discoverable without a second loader. This is a compact CPU test set. The rest of the effort belongs in the material and kernels.
