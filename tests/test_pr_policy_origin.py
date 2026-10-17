import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate


class HistoricalPolicyOriginTests(unittest.TestCase):
    def test_pinned_historical_origin_is_valid(self):
        self.assertEqual([], validate.validate_historical_policy_origin())

    def test_both_frozen_receipts_use_the_pinned_historical_origin(self):
        paths = {
            "data/pr-architecture-evidence.json": validate.PR_ARCHITECTURE_RECEIPT,
            "audit/pr-file-cap-reconstruction.json": validate.PR_CAP_RECONSTRUCTION_RECEIPT,
        }
        for relative_path, path in paths.items():
            raw = path.read_bytes()
            receipt = json.loads(raw)
            with self.subTest(receipt=relative_path):
                self.assertEqual(
                    [],
                    validate.policy_receipt_origin_errors(relative_path, receipt, raw),
                )

    def test_frozen_receipt_cannot_be_rebound_to_current_policy(self):
        current_policy_sha = hashlib.sha256(
            (ROOT / "scripts" / "pr_policy.py").read_bytes()
        ).hexdigest()
        paths = {
            "data/pr-architecture-evidence.json": validate.PR_ARCHITECTURE_RECEIPT,
            "audit/pr-file-cap-reconstruction.json": validate.PR_CAP_RECONSTRUCTION_RECEIPT,
        }
        for relative_path, path in paths.items():
            receipt = copy.deepcopy(json.loads(path.read_bytes()))
            receipt["policy_sha256"] = current_policy_sha
            rebound_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
            errors = validate.policy_receipt_origin_errors(
                relative_path, receipt, rebound_bytes
            )
            with self.subTest(receipt=relative_path):
                self.assertTrue(any("frozen historical receipt digest mismatch" in e for e in errors), errors)
                self.assertTrue(any("policy identity mismatch" in e for e in errors), errors)

    def test_unknown_policy_digest_and_unpinned_path_are_rejected(self):
        unknown = {"policy_sha256": "0" * 64}
        self.assertTrue(validate.policy_receipt_origin_errors(
            "data/pr-architecture-evidence.json", unknown, b"{}"
        ))
        self.assertTrue(validate.policy_receipt_origin_errors(
            "data/unpinned-receipt.json",
            {"policy_sha256": validate.HISTORICAL_POLICY_SHA256},
            b"{}",
        ))

    def test_current_policy_is_accepted_for_a_new_receipt(self):
        current = hashlib.sha256((ROOT / "scripts" / "pr_policy.py").read_bytes()).hexdigest()
        self.assertEqual([], validate.policy_receipt_origin_errors(
            "data/new-receipt.json", {"policy_sha256": current}, b"new receipt"
        ))

    def test_missing_origin_manifest_fails_historical_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            with patch.object(validate, "POLICY_ORIGIN_MANIFEST", missing):
                errors = validate.validate_historical_policy_origin()
        self.assertTrue(any("required historical policy origin is missing" in e for e in errors), errors)

    def test_altered_archive_and_rebound_manifest_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corrupt_archive = tmp_path / "policy.py.txt"
            corrupt_archive.write_bytes(b"not the pinned policy")
            with patch.object(validate, "HISTORICAL_POLICY_ARCHIVE", corrupt_archive):
                archive_errors = validate.validate_historical_policy_origin()
            self.assertTrue(any("archive digest mismatch" in e for e in archive_errors), archive_errors)

            rebound_manifest = json.loads(validate.POLICY_ORIGIN_MANIFEST.read_text(encoding="utf-8"))
            rebound_manifest["policy"]["sha256"] = "0" * 64
            manifest_path = tmp_path / "origin.json"
            manifest_path.write_text(json.dumps(rebound_manifest), encoding="utf-8")
            with patch.object(validate, "POLICY_ORIGIN_MANIFEST", manifest_path):
                manifest_errors = validate.validate_historical_policy_origin()
            self.assertTrue(any("historical policy origin digest mismatch" in e for e in manifest_errors), manifest_errors)

    def test_malformed_cap_rows_do_not_match_historical_identity(self):
        receipt = json.loads(validate.PR_CAP_RECONSTRUCTION_RECEIPT.read_bytes())
        receipt["rows"] = ["invalid row", {"repo": 1, "pr": True}]
        self.assertFalse(validate.historical_receipt_identity_matches(
            "audit/pr-file-cap-reconstruction.json", receipt
        ))

    def test_changed_pr_page_corpus_fails_historical_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = Path(tmp) / "sources"
            page = sources / "prs" / "example" / "PR-1.md"
            page.parent.mkdir(parents=True)
            page.write_text("changed source page\n", encoding="utf-8")
            with patch.object(validate, "SOURCES_DIR", sources):
                errors = validate.validate_historical_policy_origin()
        self.assertTrue(any("sources/prs historical corpus digest mismatch" in e for e in errors), errors)

    def test_altered_frozen_receipt_fails_historical_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            altered = Path(tmp) / "pr-architecture-evidence.json"
            altered.write_bytes(validate.PR_ARCHITECTURE_RECEIPT.read_bytes() + b" ")
            with patch.object(validate, "PR_ARCHITECTURE_RECEIPT", altered):
                errors = validate.validate_historical_policy_origin()
        self.assertTrue(any("data/pr-architecture-evidence.json: frozen historical receipt digest mismatch" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
