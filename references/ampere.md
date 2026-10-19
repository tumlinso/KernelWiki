# Ampere / A100 routing

Use exact `sm80` guidance for an A100-class target. The `ampere` family query
also returns other SM 8.x evidence; resource limits and product-level values can
differ within that family.

    python3 scripts/query.py --architecture A100 --type hardware --compact
    python3 scripts/query.py --architecture sm80 --type hardware --compact
    python3 scripts/query.py --architecture ampere --type hardware --compact
    python3 scripts/get_page.py hw-ampere-async-copy --follow-sources
    python3 scripts/get_page.py hw-ampere-tensor-operations --follow-sources
    python3 scripts/get_page.py hw-ampere-memory-resources --follow-sources

| Decision | Page |
|---|---|
| Is global-to-shared staging asynchronous in hardware? | `hw-ampere-async-copy` |
| Which dense and structured-sparse Tensor Core forms are eligible, and what are their accumulation types? | `hw-ampere-tensor-operations` |
| What memory/cache/resource budgets apply to SM80/A100? | `hw-ampere-memory-resources` |

## Scope boundary

The Ampere Tuning Guide and CUDA 12.9.1 Programming Guide are the primary
sources for these contracts. The pages are source-reported and make no local
performance claim. Check the exact compiler target and generated binary for a
custom PTX path. An API name can be available on an older target while its
hardware acceleration begins at SM80. Later TMA, warp-group MMA, cluster, and
Blackwell mechanisms have separate architecture pages and are not Ampere
synonyms.
