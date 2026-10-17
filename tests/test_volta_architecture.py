import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import query  # noqa: E402
from pr_policy import (  # noqa: E402
    SUPPORTED_EXACT_ARCHITECTURES,
    architecture_matches_filter,
    derive_architectures,
)


def load_hyphenated(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


indices = load_hyphenated("generate_indices_volta", SCRIPTS / "generate-indices.py")


def changed(path, patch_text=""):
    return {"filename": path, "patch": patch_text}


def page(path, architectures, disposition=None):
    frontmatter = {"architectures": architectures}
    if disposition is not None:
        frontmatter["architecture_disposition"] = disposition
    return {**frontmatter, "path": path, "fm": frontmatter}


class VoltaArchitectureTests(unittest.TestCase):
    def test_exact_volta_targets_and_product_evidence(self):
        self.assertIn("sm70", SUPPORTED_EXACT_ARCHITECTURES)
        self.assertIn("sm72", SUPPORTED_EXACT_ARCHITECTURES)
        for term in ("V100", "GV100", "sm70", "sm_70"):
            with self.subTest(term=term):
                architectures, disposition, evidence = derive_architectures(
                    f"{term} GEMM", "", []
                )
                self.assertEqual(["sm70"], architectures)
                self.assertEqual("exact", disposition)
                if term.lower() in {"v100", "gv100"}:
                    self.assertTrue(all(
                        row.get("mapping_source")
                        == "https://developer.nvidia.com/cuda/gpus/legacy"
                        for row in evidence
                    ))

    def test_generic_volta_and_neighboring_sm72_remain_distinct(self):
        family = derive_architectures("Volta tuning", "", [])
        neighbor = derive_architectures("SM72 target", "", [])
        self.assertEqual((["volta"], "family"), family[:2])
        self.assertEqual((["sm72"], "exact"), neighbor[:2])

        self.assertTrue(architecture_matches_filter(["sm70"], "exact", "V100"))
        self.assertTrue(architecture_matches_filter(["sm70"], "exact", "sm_70"))
        self.assertFalse(architecture_matches_filter(["sm72"], "exact", "sm70"))
        self.assertFalse(architecture_matches_filter(["sm70"], "exact", "sm72"))
        self.assertTrue(architecture_matches_filter(["volta"], "family", "volta"))
        self.assertTrue(architecture_matches_filter(["sm70"], "exact", "volta"))
        self.assertTrue(architecture_matches_filter(["sm72"], "exact", "volta"))

    def test_numeric_architecture_guards_cover_sm70_and_sm72(self):
        for guard, expected in (
            ("__CUDA_ARCH__ == 700", "sm70"),
            ("__CUDA_ARCH__ == 720", "sm72"),
            ("get_sm_version() == 70", "sm70"),
            ("compute_capability >= 72", "sm72"),
        ):
            with self.subTest(guard=guard):
                result = derive_architectures(
                    "Kernel dispatch", "", [changed("src/kernel.cu", f"+{guard}")]
                )
                self.assertEqual([expected], result[0])
                self.assertEqual("exact", result[1])

    def test_unknown_and_unsupported_sm70_suffixes_do_not_gain_volta_attribution(self):
        unknown = derive_architectures("Generic kernel", "", [changed("kernel.cu")])
        self.assertEqual(([], "unknown"), unknown[:2])
        for token in ("sm70a", "sm70f", "sm700"):
            with self.subTest(token=token):
                result = derive_architectures(
                    f"{token} dispatch", "", [changed("kernel.cu")]
                )
                self.assertEqual(([], "unknown"), result[:2])
                self.assertFalse(architecture_matches_filter([token], "exact", "volta"))
        self.assertFalse(architecture_matches_filter(["sm700"], "exact", "volta"))

    def test_unknown_filter_stays_explicit(self):
        self.assertTrue(architecture_matches_filter([], "unknown", "unknown"))
        self.assertFalse(architecture_matches_filter([], "", "unknown"))
        self.assertFalse(architecture_matches_filter(["sm70"], "unknown", "unknown"))

    def test_hopper_and_blackwell_controls_keep_their_hierarchy(self):
        self.assertTrue(architecture_matches_filter(["sm90"], "exact", "H100"))
        self.assertTrue(architecture_matches_filter(["sm90"], "exact", "hopper"))
        self.assertTrue(architecture_matches_filter(["sm100a"], "exact", "blackwell"))
        self.assertFalse(architecture_matches_filter(["sm100"], "exact", "sm90"))

    def test_query_diagnostic_uses_the_shared_family_for_exact_v100(self):
        pages = [
            {"fm": {"architectures": ["volta"], "architecture_disposition": "family"}},
            {"fm": {"architectures": [], "architecture_disposition": "unknown"}},
            {"fm": {"architectures": ["blackwell"], "architecture_disposition": "family"}},
        ]
        self.assertIsNone(query.architecture_filter_diagnostic(pages, "volta"))
        diagnostic = query.architecture_filter_diagnostic(pages, "V100")
        self.assertIn("family-level evidence", diagnostic)
        self.assertIn("--architecture volta", diagnostic)
        self.assertNotIn("--architecture blackwell", diagnostic)
        self.assertIn("architecture-unknown page(s)", diagnostic)

    def test_alias_data_keeps_volta_family_separate_from_v100(self):
        query._ALIAS_CACHE = None
        aliases = query.load_alias_expansions()
        self.assertEqual("volta", aliases["volta"])
        for alias in ("v100", "gv100", "sm_70"):
            self.assertEqual("sm70", aliases[alias])
        self.assertEqual("sm72", aliases["sm_72"])

    def test_architecture_index_matches_exact_family_and_unknown_queries(self):
        pages = [
            page("v100", ["sm70"], "exact"),
            page("xavier", ["sm72"], "exact"),
            page("volta-family", ["volta"], "family"),
            page("unknown", [], "unknown"),
        ]
        exact, families, unknown = indices.architecture_index_sets(pages)
        self.assertEqual({"v100"}, {p["path"] for p in exact["sm70"]})
        self.assertEqual({"xavier"}, {p["path"] for p in exact["sm72"]})
        self.assertEqual({"volta-family"}, {p["path"] for p in families["volta"]})
        self.assertEqual({"unknown"}, {p["path"] for p in unknown})
        self.assertEqual(
            {"v100"},
            {p["path"] for p in pages if architecture_matches_filter(
                p["fm"].get("architectures", []),
                p["fm"].get("architecture_disposition", ""), "sm70"
            )},
        )
        self.assertEqual(
            {"v100", "xavier", "volta-family"},
            {p["path"] for p in pages if architecture_matches_filter(
                p["fm"].get("architectures", []),
                p["fm"].get("architecture_disposition", ""), "volta"
            )},
        )

    def test_multiple_language_guides_are_sorted_and_single_guide_output_is_unchanged(self):
        guides = [
            {"id": "lang-cuda-cpp", "title": "CUDA C++ guide", "type": "language",
             "tags": ["cuda-cpp"], "_path": "wiki/languages/cuda-cpp.md"},
            {"id": "lang-volta-cuda", "title": "Volta CUDA C++ guide", "type": "language",
             "tags": ["cuda-cpp"], "_path": "wiki/languages/volta-cuda-cpp.md"},
            {"id": "lang-ptx", "title": "PTX guide", "type": "language",
             "tags": ["ptx"], "_path": "wiki/languages/ptx.md"},
            {"id": "lang-volta-ptx", "title": "Volta PTX guide", "type": "language",
             "tags": ["ptx"], "_path": "wiki/languages/volta-ptx.md"},
        ]
        with patch.object(indices, "load_valid_languages", return_value={"cuda-cpp", "ptx"}):
            rendered = indices.generate_by_language(guides)
            reversed_rendered = indices.generate_by_language(list(reversed(guides)))
            self.assertEqual(rendered, reversed_rendered)
            for link in (
                "[CUDA C++ guide](../wiki/languages/cuda-cpp.md)",
                "[Volta CUDA C++ guide](../wiki/languages/volta-cuda-cpp.md)",
                "[PTX guide](../wiki/languages/ptx.md)",
                "[Volta PTX guide](../wiki/languages/volta-ptx.md)",
            ):
                self.assertIn(link, rendered)

            single = indices.generate_by_language(guides[:1])
            self.assertIn(
                "| `cuda-cpp` | [CUDA C++ guide](../wiki/languages/cuda-cpp.md) |  |",
                single,
            )


if __name__ == "__main__":
    unittest.main()
