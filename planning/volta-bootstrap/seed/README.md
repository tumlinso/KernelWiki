# Prepared additive material

This subtree contains native-style source records, six concept-level hardware pages, and a Volta routing reference. It is deliberately not a collection of uncompiled kernels masquerading as validated examples.

`ADOPTION.json` lists the small vocabulary additions needed before native validation. The implementing agent should review and copy individual additions into the corresponding canonical paths, not overwrite existing files blindly. Source identifiers are intentionally Volta-specific; existing equivalent source IDs may be reused instead if the native corpus already has an exact matching record.

The drafts were checked for YAML shape, IDs and internal references during package preparation. Actual KernelWiki validation after adoption, tool-version registry checks, compilation, GPU correctness and measurements remain implementation tasks. Do not raise confidence or reproducibility merely because the package checker passes.
