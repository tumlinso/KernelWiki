import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bundle import (
    HEADER,
    INPUT_MAGIC,
    ORACLE_MAGIC,
    generate_bundle,
    read_input,
    read_oracle,
    verify_bundle,
)
from suite_records import verify_suite_records


class BinaryBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[2] / 'cases' / 'fixture_bundle'
        manifest_path = cls.root / 'bundle-manifest.json'
        # Reuse the already generated deterministic CPU bundle when present.
        # `bundle.py generate` remains the explicit regeneration entry point;
        # avoiding regeneration here prevents every reader/corruption test
        # from paying for the weighted FP64 reference construction again.
        if manifest_path.is_file():
            cls.generated = json.loads(manifest_path.read_text())
        else:
            cls.generated = generate_bundle(cls.root)
        cls.verified = verify_bundle(cls.root)

    def copied_bundle(self, temp):
        target = Path(temp) / 'bundle'
        shutil.copytree(self.root, target)
        return target

    def test_generated_bundle_roundtrips_all_represented_inputs_and_fp64_oracles(self):
        root = self.root
        generated = self.generated
        result = self.verified
        self.assertEqual(result['cases'], 10)
        self.assertEqual(len(generated['cases']), 10)
        self.assertEqual(set(result['case_ids']), {
            'tiny_balanced', 'tiny_skewed', 'moderate_balanced',
            'moderate_skewed', 'weighted_balanced', 'weighted_skewed',
            'low_m', 'moderate_m', 'weighted_low_m', 'weighted_moderate_m'})
        for case in generated['cases']:
            inp = read_input(root / case['input']['path'])
            oracle = read_oracle(root / case['oracle']['path'])
            self.assertEqual(inp['kind'], oracle['kind'])
            contract = json.loads((root / case['contract']['path']).read_text())
            self.assertEqual(contract['input_sha256'], case['input']['sha256'])
            self.assertEqual(contract['oracle_sha256'], case['oracle']['sha256'])
            self.assertTrue(all(value == value for value in oracle['y64']))
            self.assertLess(case['input']['bytes'], 80 * 1024 * 1024)
            self.assertLess(case['oracle']['bytes'], 80 * 1024 * 1024)

    def test_grouped_binary_header_and_output_reference_extents(self):
        root = self.root
        record = next(c for c in json.loads((root / 'bundle-manifest.json').read_text())['cases']
                      if c['id'] == 'tiny_skewed')
        raw = (root / record['input']['path']).read_bytes()
        header = HEADER.unpack_from(raw)
        self.assertEqual(header[0], INPUT_MAGIC)
        self.assertEqual(header[1], 1)
        parsed = read_input(root / record['input']['path'])
        self.assertEqual(len(parsed['assignments']), 8)
        oracle = read_oracle(root / record['oracle']['path'])
        self.assertEqual(len(oracle['y64']), 8 * parsed['n'])
        self.assertEqual(len(oracle['sum_abs_up']), len(oracle['y64']))
        self.assertEqual(len(oracle['sum_abs_gate']), len(oracle['y64']))
        self.assertEqual(len(oracle['up_reference']), len(oracle['y64']))
        self.assertEqual(len(oracle['gate_reference']), len(oracle['y64']))

    def test_w4_binary_preserves_odd_code_tail_and_scales(self):
        root = self.root
        record = next(c for c in json.loads((root / 'bundle-manifest.json').read_text())['cases']
                      if c['id'] == 'low_m')
        parsed = read_input(root / record['input']['path'])
        count = parsed['k'] * parsed['n']
        self.assertEqual(count % 2, 1)  # Native bundle also exercises ignored odd tail nibble.
        self.assertEqual(len(parsed['packed_codes']), (count + 1) // 2)
        self.assertEqual(len(parsed['scales']), 3 * parsed['n'])
        self.assertLess(min(parsed['activation16']), 0.0)
        self.assertGreater(max(parsed['activation16']), 0.0)
        self.assertEqual(len(read_oracle(root / record['oracle']['path'])['y64']), parsed['m'] * parsed['n'])

    def test_corrupt_or_truncated_input_and_oracle_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='kwv07-bundle-') as temp:
            root = self.copied_bundle(temp)
            record = next(c for c in json.loads((root / 'bundle-manifest.json').read_text())['cases']
                          if c['id'] == 'low_m')
            input_path = root / record['input']['path']
            original_input = input_path.read_bytes()
            input_path.write_bytes(original_input[:-1])
            with self.assertRaises(ValueError):
                read_input(input_path)
            input_path.write_bytes(original_input)
            oracle_path = root / record['oracle']['path']
            original_oracle = oracle_path.read_bytes()
            oracle_path.write_bytes(original_oracle + b'\x00')
            with self.assertRaises(ValueError):
                read_oracle(oracle_path)

    def test_wrong_magic_and_trailing_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='kwv07-bundle-') as temp:
            root = self.copied_bundle(temp)
            record = next(c for c in json.loads((root / 'bundle-manifest.json').read_text())['cases']
                          if c['id'] == 'tiny_balanced')
            path = root / record['input']['path']
            original = path.read_bytes()
            path.write_bytes(b'BADMAGIC' + original[8:])
            with self.assertRaises(ValueError):
                read_input(path)
            path.write_bytes(original + b'\x00')
            with self.assertRaises(ValueError):
                read_input(path)

    def test_manifest_hash_checks_detect_modified_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='kwv07-bundle-') as temp:
            root = self.copied_bundle(temp)
            record = json.loads((root / 'bundle-manifest.json').read_text())['cases'][0]
            input_path = root / record['input']['path']
            input_path.write_bytes(input_path.read_bytes() + b'\x00')
            with self.assertRaises(ValueError):
                verify_bundle(root)

    def test_suite_wrappers_bind_exact_members_and_are_explicitly_cpu_only(self):
        root = self.root
        manifest = json.loads((root / 'bundle-manifest.json').read_text())
        grouped_contract = json.loads((root / 'contracts/grouped-suite.json').read_text())
        grouped_oracle = json.loads((root / 'oracles/grouped-suite.json').read_text())
        w4_contract = json.loads((root / 'contracts/w4a16-suite.json').read_text())
        w4_oracle = json.loads((root / 'oracles/w4a16-suite.json').read_text())
        self.assertEqual(grouped_contract['case_id'], 'kwv07-grouped-gated')
        self.assertEqual(w4_contract['case_id'], 'kwv07-w4a16')
        self.assertEqual(grouped_contract['case_ids'], [
            'tiny_balanced', 'tiny_skewed', 'moderate_balanced', 'moderate_skewed',
            'weighted_balanced', 'weighted_skewed'])
        self.assertEqual(w4_contract['case_ids'], [
            'low_m', 'moderate_m', 'weighted_low_m', 'weighted_moderate_m'])
        self.assertEqual(len(grouped_oracle['checks']), 6)
        self.assertEqual(len(w4_oracle['checks']), 4)
        self.assertIn('no GPU execution', grouped_oracle['validation_scope'])
        self.assertIn('no GPU execution', w4_oracle['validation_scope'])
        for contract, oracle in (
            (grouped_contract, grouped_oracle),
            (w4_contract, w4_oracle),
        ):
            suite_ref = contract['suite_manifest']
            suite_path = root / suite_ref['path']
            self.assertEqual(suite_path.parent, root / 'suites')
            self.assertEqual(suite_ref['sha256'], hashlib.sha256(suite_path.read_bytes()).hexdigest())
            self.assertEqual(oracle['suite_manifest'], suite_ref)
            suite_manifest = json.loads(suite_path.read_text())
            for case in suite_manifest['cases']:
                for artifact in (case['input'], case['oracle'], case['contract']):
                    self.assertEqual(artifact['sha256'], hashlib.sha256((root / artifact['path']).read_bytes()).hexdigest())
        self.assertTrue(all(check['result'] == 'pass' and check['atol'] >= check['max_abs_error']
                            for check in grouped_oracle['checks'] + w4_oracle['checks']))
        expected_ids = {case['id'] for case in manifest['cases']}
        self.assertEqual(
            {entry['id'] for entry in json.loads((root / 'suites/grouped.json').read_text())['cases']}
            | {entry['id'] for entry in json.loads((root / 'suites/w4a16.json').read_text())['cases']},
            expected_ids)
        self.assertEqual(self.verified['cases'], 10)

    def test_suite_wrapper_corruption_is_detected_without_rewriting(self):
        with tempfile.TemporaryDirectory(prefix='kwv07-bundle-') as temp:
            root = self.copied_bundle(temp)
            wrapper_path = root / 'oracles/grouped-suite.json'
            original = wrapper_path.read_bytes()
            wrapper_path.write_bytes(original + b' ')
            with self.assertRaisesRegex(ValueError, 'suite wrapper identity/precheck mismatch'):
                verify_suite_records(root, json.loads((root / 'bundle-manifest.json').read_text()))
            self.assertEqual(wrapper_path.read_bytes(), original + b' ')

    def test_weighted_families_share_exact_represented_weights(self):
        manifest = json.loads((self.root / 'bundle-manifest.json').read_text())
        by_id = {case['id']: case for case in manifest['cases']}
        weighted_grouped = [read_input(self.root / by_id[name]['input']['path'])
                            for name in ('weighted_balanced', 'weighted_skewed')]
        self.assertEqual(weighted_grouped[0]['x16'], weighted_grouped[1]['x16'])
        self.assertEqual(weighted_grouped[0]['up16'], weighted_grouped[1]['up16'])
        self.assertEqual(weighted_grouped[0]['gate16'], weighted_grouped[1]['gate16'])
        weighted_w4 = [read_input(self.root / by_id[name]['input']['path'])
                       for name in ('weighted_low_m', 'weighted_moderate_m')]
        self.assertEqual(weighted_w4[0]['k'], 4096)
        self.assertEqual(weighted_w4[0]['n'], 1024)
        self.assertEqual(weighted_w4[0]['packed_codes'], weighted_w4[1]['packed_codes'])
        self.assertEqual(weighted_w4[0]['scales'], weighted_w4[1]['scales'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
