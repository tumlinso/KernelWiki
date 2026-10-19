"""Regression checks for the pinned Atlas campaign evidence routes."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import query  # noqa: E402


PAGE_IDS = {
    "hw-volta-atlas-evidence-index",
    "hw-volta-atlas-e09-mma-coordinate",
    "hw-volta-atlas-e11-packed-intersection",
    "hw-volta-atlas-e12-independent-transform",
    "hw-volta-atlas-e16-resident-scheduling",
    "hw-volta-atlas-e23-vmm-aliasing",
    "hw-volta-atlas-e39-bitplane-addition",
}
SOURCE_ID = "blog-atlas-v100-campaign-20261006"
ATLAS_SHA = "5b5629b001da4a42d6f41a84d94f73c07a29d478"


def filter_args(architecture, confidence=None):
    return SimpleNamespace(
        type="hardware",
        tag=None,
        repo=None,
        language=None,
        architecture=architecture,
        symptom=None,
        confidence=confidence,
        has_code=False,
    )


class AtlasEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = query.load_all_pages()
        cls.by_id = {p["fm"].get("id"): p for p in cls.pages}

    def test_source_and_observation_pages_are_pinned_and_source_reported(self):
        source_path = ROOT / "sources/blogs/atlas-v100-campaign-20261006.md"
        source, _ = query.load_frontmatter(source_path)
        self.assertEqual(SOURCE_ID, source["id"])
        self.assertIn(ATLAS_SHA, source["url"])
        self.assertEqual("benchmark-blog", source["source_category"])
        self.assertEqual(["sm70"], source["architectures"])
        source_text = source_path.read_text(encoding="utf-8")
        self.assertIn("access-gated", source_text)
        self.assertIn("git -C <atlas-checkout> show", source_text)
        self.assertNotIn("/home/tumlinson/", source_text)

        self.assertTrue(PAGE_IDS.issubset(self.by_id))
        for page_id in PAGE_IDS:
            fm = self.by_id[page_id]["fm"]
            self.assertEqual("hardware", fm["type"])
            self.assertEqual(["sm70"], fm["architectures"])
            self.assertEqual("source-reported", fm["confidence"])
            self.assertNotIn("performance_claims", fm)
            self.assertIn(SOURCE_ID, fm["sources"])

    def test_exact_architecture_and_confidence_retrieval(self):
        exact_sm70 = {
            p["fm"]["id"]
            for p in query.filter_pages(self.pages, filter_args("sm70", "source-reported"))
        }
        self.assertTrue(PAGE_IDS.issubset(exact_sm70))

        sm72_only = {
            p["fm"]["id"]
            for p in query.filter_pages(self.pages, filter_args("sm72", "source-reported"))
        }
        self.assertTrue(PAGE_IDS.isdisjoint(sm72_only))

    def test_key_limitations_and_restricted_non_results_remain_retrievable(self):
        expected = {
            "hw-volta-atlas-e09-mma-coordinate": "did not check which lane owns a slot",
            "hw-volta-atlas-e11-packed-intersection": "actual iteration count to one",
            "hw-volta-atlas-e12-independent-transform": "did not feed the prior output back",
            "hw-volta-atlas-e16-resident-scheduling": "not a finding that persistent execution is generally slower",
            "hw-volta-atlas-e23-vmm-aliasing": "does not isolate modulo as the cause",
            "hw-volta-atlas-e39-bitplane-addition": "no statistical crossover",
        }
        for page_id, phrase in expected.items():
            with self.subTest(page=page_id):
                body = " ".join(self.by_id[page_id]["body"].split())
                self.assertIn(phrase, body)

        index = self.by_id["hw-volta-atlas-evidence-index"]["body"]
        for experiment in ("E27", "E28", "E29", "E30"):
            self.assertIn(f"| {experiment} | NOT_RUN |", index)
        self.assertIn("zero cases", index)
        self.assertIn("do not validate", index)


if __name__ == "__main__":
    unittest.main()
