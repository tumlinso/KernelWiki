import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import query  # noqa: E402


def query_args(architecture, confidence="source-reported"):
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


class AtlasContractRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = query.load_all_pages()

    def hardware_ids(self, architecture):
        result = query.filter_pages(
            self.pages,
            query_args(architecture),
        )
        return {
            page["fm"]["id"]
            for page in result
            if page["_ptype"] == "wiki-hardware"
        }

    def test_sm80_and_a100_queries_retrieve_only_sm80_contract_pages(self):
        expected = {
            "hw-ampere-async-copy",
            "hw-ampere-tensor-operations",
            "hw-ampere-memory-resources",
        }
        self.assertTrue(expected.issubset(self.hardware_ids("sm80")))
        self.assertTrue(expected.issubset(self.hardware_ids("A100")))
        self.assertTrue(expected.issubset(self.hardware_ids("ampere")))
        self.assertTrue(expected.isdisjoint(self.hardware_ids("sm86")))
        self.assertTrue(expected.isdisjoint(self.hardware_ids("sm70")))

    def test_volta_integer_aliases_route_to_the_exact_contract_page(self):
        args = query_args(None)
        args.tag = "__byte_perm"
        matches = query.filter_pages(self.pages, args)
        hardware = {
            page["fm"]["id"]
            for page in matches
            if page["_ptype"] == "wiki-hardware"
        }
        self.assertIn("hw-volta-integer-routing", hardware)
        self.assertTrue(hardware.isdisjoint(self.hardware_ids("sm80")))

    def test_sm70_mma_and_vmm_contracts_stay_exact(self):
        ids = self.hardware_ids("V100")
        self.assertIn("hw-volta-mma", ids)
        self.assertIn("hw-volta-vmm-addressing", ids)
        self.assertIn("hw-volta-integer-routing", ids)
        self.assertNotIn("hw-ampere-async-copy", ids)

        for page_id, required_source in (
            ("hw-volta-mma", "doc-volta-ptx-8-8"),
            ("hw-volta-vmm-addressing", "doc-volta-cuda-12-9"),
            ("hw-volta-integer-routing", "doc-volta-math-12-9"),
        ):
            with self.subTest(page_id=page_id):
                page = next(page for page in self.pages if page["fm"].get("id") == page_id)
                self.assertEqual(["sm70"], page["fm"]["architectures"])
                self.assertEqual("source-reported", page["fm"]["confidence"])
                self.assertIn(required_source, page["fm"]["sources"])

    def test_contract_pages_retain_specificity_and_experiment_limits(self):
        mma = (ROOT / "wiki/hardware/volta-mma.md").read_text(encoding="utf-8")
        self.assertIn("(l mod 4) + 4·[l ≥ 16]", mma)
        self.assertIn("raw WMMA ownership", mma)
        self.assertIn("volta-atlas-e09-mma-coordinate.md", mma)

        vmm = (ROOT / "wiki/hardware/volta-vmm-addressing.md").read_text(encoding="utf-8")
        self.assertIn("cuMemSetAccess", vmm)
        self.assertIn("fence.proxy.alias", vmm)
        self.assertIn("volta-atlas-e23-vmm-aliasing.md", vmm)
        self.assertIn("did not isolate modulo as the cause", vmm)

        async_copy = (ROOT / "wiki/hardware/ampere-async-copy.md").read_text(encoding="utf-8")
        self.assertIn("hardware-accelerated asynchronous global-to-shared copies", async_copy)
        self.assertIn("`.ca` allows caching at all levels including L1", async_copy)
        self.assertIn("`.cg` caches at L2 and bypasses L1", async_copy)
        self.assertIn("not ready for an ordinary shared-memory load", async_copy)
        self.assertIn("This page is scoped to SM80", async_copy)

        integer = (ROOT / "wiki/hardware/volta-integer-routing.md").read_text(encoding="utf-8")
        self.assertIn("requires `sm_61` or later", integer)
        self.assertIn("related PTX `dp2a` form also requires `sm_61` or later", integer)
        self.assertIn("PTX ISA 8.2", integer)
        self.assertIn("0xffffffff", integer)
        self.assertIn("do not imply a memory barrier", integer)

        tensor = (ROOT / "wiki/hardware/ampere-tensor-operations.md").read_text(encoding="utf-8")
        self.assertIn("supports `mma.sp` structured-sparse forms", tensor)
        self.assertIn("Sparse TF32 instead selects one position from each pair", tensor)
        self.assertIn("metadata operand", tensor)
        self.assertIn("selected payload value may itself be numerically zero", tensor)
        self.assertIn("cannot be represented losslessly", tensor)
        self.assertIn("does not discover or compress arbitrary zeros", tensor)


if __name__ == "__main__":
    unittest.main()
