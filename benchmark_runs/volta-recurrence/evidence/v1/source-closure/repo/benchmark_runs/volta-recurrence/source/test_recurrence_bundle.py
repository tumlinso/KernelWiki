import unittest

import numpy as np

from bundle_io import BundleCase, decode_bundle, encode_bundle
from prepare_recurrence_bundle import make_case
from recurrence_oracle import run_interleaved


class RecurrenceBundleTests(unittest.TestCase):
    def test_represented_inputs_and_fp64_references_round_trip(self):
        case = make_case("tiny", 9, 5, 3, 4, 0x808)
        binary, manifest = encode_bundle([case])
        restored, decoded_manifest = decode_bundle(binary)
        self.assertEqual(manifest["bundle_file_sha256"],
                         decoded_manifest["bundle_file_sha256"])
        self.assertEqual(len(restored), 1)
        actual = restored[0]
        self.assertEqual(actual.case_id, case.case_id)
        for name in ("keys", "values", "queries", "alpha", "beta", "lengths",
                     "initial_states", "expected_outputs", "expected_final_states"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(case, name))

    def test_payload_corruption_and_trailing_bytes_are_rejected(self):
        case = make_case("tiny", 4, 3, 2, 3, 0x809)
        binary, _ = encode_bundle([case])
        damaged = bytearray(binary)
        damaged[-1] ^= 0x01
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            decode_bundle(bytes(damaged))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            decode_bundle(binary + b"x")

    def test_padding_is_marked_and_live_reference_is_independently_recomputable(self):
        case = make_case("ragged", 12, 6, 3, 5, 0x80A)
        live = np.arange(case.shape[0])[:, None] < case.lengths[None, :]
        self.assertTrue(np.isnan(case.keys[~live]).all())
        self.assertTrue(np.isnan(case.expected_outputs[~live]).all())
        outputs, final = run_interleaved(
            case.keys, case.values, case.queries, case.alpha, case.beta,
            case.initial_states, case.lengths
        )
        np.testing.assert_array_equal(outputs, case.expected_outputs)
        np.testing.assert_array_equal(final, case.expected_final_states)
        self.assertTrue(np.all(case.initial_states[0] != 0))
        np.testing.assert_array_equal(final[0], case.initial_states[0].astype(np.float64))

    def test_numeric_policy_rejects_a_real_wrong_output(self):
        case = make_case("tiny", 3, 2, 2, 3, 0x80B)
        outputs, _ = run_interleaved(
            case.keys, case.values, case.queries, case.alpha, case.beta,
            case.initial_states, case.lengths
        )
        live = np.arange(case.shape[0])[:, None] < case.lengths[None, :]
        wrong = outputs.copy()
        first = np.argwhere(live)[0]
        wrong[first[0], first[1], 0] += 1.0
        allowed = 2.0e-5 + 2.0e-4 * np.abs(outputs[live])
        self.assertTrue(np.any(np.abs(wrong[live] - outputs[live]) > allowed))


if __name__ == "__main__":
    unittest.main()
