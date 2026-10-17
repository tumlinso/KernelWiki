import unittest
import struct

from prepare_rows import performance_cases, route_manifest, suites
from row_oracle import (ATOL, RTOL, Case, ContractError, check_output,
                       compute_reference, decode_bundle, decode_reference,
                       encode_bundle, encode_reference, oracle_record)


class RowContractTests(unittest.TestCase):
    def test_bundle_and_reference_are_deterministic_and_represented(self):
        for operation, cases in suites().items():
            encoded = encode_bundle(cases)
            decoded = decode_bundle(encoded)
            refs = [(case.case_id, compute_reference(case)) for case in cases]
            reference_bytes = encode_reference(refs)
            self.assertEqual(encoded, encode_bundle(cases))
            self.assertEqual(reference_bytes,
                             encode_reference([(case.case_id, compute_reference(case))
                                               for case, _, _ in decoded]))
            self.assertEqual(set(decode_reference(reference_bytes)),
                             {case.case_id for case in cases})
            record = oracle_record(encoded, reference_bytes)
            self.assertTrue(record["passed"])
            self.assertEqual(record["checks"][0]["atol"], ATOL)
            self.assertEqual(record["checks"][0]["rtol"], RTOL)

    def test_fixed_mixed_operator_suite_is_one_oracle_bundle(self):
        cases = [case for group in suites().values() for case in group]
        encoded = encode_bundle(cases)
        reference = encode_reference([(case.case_id, compute_reference(case)) for case in cases])
        record = oracle_record(encoded, reference)
        self.assertTrue(record["passed"])
        self.assertEqual(len(record["checks"]), len(cases))
        self.assertEqual({item["operation"] for item in record["checks"]},
                         {"softmax", "rmsnorm"})

    def test_contract_routes_bind_actual_alignment_fallbacks_and_empty_rows(self):
        all_cases = [case for group in suites().values() for case in group] + performance_cases()
        manifest = {route["case_id"]: route for case in all_cases
                    for route in [route_manifest(case)]}
        self.assertEqual(manifest["softmax_n33"]["baseline_path"], "softmax_warp")
        self.assertEqual(manifest["softmax_n33"]["candidate_path"], "softmax_cta")
        self.assertEqual(manifest["rmsnorm_n33"]["candidate_path"], "vec4")
        self.assertEqual(manifest["rmsnorm_n33_misaligned"]["candidate_path"],
                         "scalar_fallback")
        self.assertEqual(manifest["rmsnorm_zero_rows"]["candidate_path"], "no_launch")
        self.assertEqual(manifest["softmax_wide_streaming"]["expected_outputs"], 512 * 4096)
        self.assertEqual(manifest["rmsnorm_vec4_wide_streaming"]["candidate_path"], "vec4")
        self.assertEqual(manifest["rmsnorm_n33_misaligned"]["check_identity"],
                         "logical-row-major:rmsnorm_n33_misaligned")

    def test_mask_tail_stride_and_all_masked_row(self):
        case = next(c for c in suites()["softmax"] if c.case_id == "softmax_mask_tail_stride")
        decoded = decode_bundle(encode_bundle([case]))[0][0]
        self.assertEqual((decoded.rows, decoded.cols), (5, 33))
        self.assertEqual((decoded.x_offset, decoded.y_offset, decoded.mask_offset), (1, 3, 2))
        ref = compute_reference(decoded)
        self.assertEqual(ref[-decoded.cols:], [0.0] * decoded.cols)
        for row in range(decoded.rows - 1):
            self.assertAlmostEqual(sum(ref[row * decoded.cols:(row + 1) * decoded.cols]), 1.0,
                                   places=12)

    def test_rmsnorm_oracle_uses_fp64_on_decoded_fp32(self):
        case = next(c for c in suites()["rmsnorm"] if c.case_id == "rmsnorm_n33_misaligned")
        decoded = decode_bundle(encode_bundle([case]))[0][0]
        expected = compute_reference(decoded)
        self.assertEqual(len(expected), decoded.rows * decoded.cols)
        self.assertTrue(all(abs(value) <= 4.0 * (decoded.cols ** 0.5) + 1e-5
                            for value in expected))

    def test_zero_rows_has_positive_width_and_empty_reference(self):
        for op in ("softmax", "rmsnorm"):
            case = next(c for c in suites()[op] if c.rows == 0)
            self.assertGreaterEqual(case.cols, 1)
            self.assertEqual(compute_reference(case), [])
            self.assertEqual(decode_bundle(encode_bundle([case]))[0][0].rows, 0)

    def test_width_one_is_valid_but_width_zero_is_not(self):
        softmax = Case("softmax_n1", "softmax", 1, 1, 1, 1, 1,
                       0, 0, 0, 0, 0.0, [1.25], [], [])
        rmsnorm = Case("rmsnorm_n1", "rmsnorm", 1, 1, 1, 1, 1,
                       0, 0, 0, 0, 1.0e-4, [2.0], [], [0.5])
        self.assertEqual(compute_reference(decode_bundle(encode_bundle([softmax]))[0][0]), [1.0])
        self.assertAlmostEqual(compute_reference(rmsnorm)[0],
                               1.0 / ((4.0 + 1.0e-4) ** 0.5), places=12)
        invalid = Case("softmax_n0", "softmax", 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0.0, [], [], [])
        with self.assertRaises(ContractError):
            encode_bundle([invalid])
        oversized = Case("softmax_n8194", "softmax", 0, 8194, 8194, 8194, 8194,
                         0, 0, 0, 0, 0.0, [], [], [])
        with self.assertRaises(ContractError):
            encode_bundle([oversized])

    def test_finite_domain_and_layout_are_enforced(self):
        bad = Case("bad", "softmax", 1, 2, 2, 2, 2, 0, 0, 0, 0, 0.0,
                   [float("nan"), 0.0], [], [])
        with self.assertRaises(ContractError):
            encode_bundle([bad])
        bad_epsilon = Case("bad_eps", "rmsnorm", 1, 1, 1, 1, 1, 0, 0, 0, 0,
                           0.0, [0.0], [], [1.0])
        with self.assertRaises(ContractError):
            encode_bundle([bad_epsilon])

    def test_reader_rejects_truncation_and_trailing_data(self):
        raw = encode_bundle([suites()["softmax"][0]])
        with self.assertRaises(ContractError):
            decode_bundle(raw[:-1])
        with self.assertRaises(ContractError):
            decode_bundle(raw + b"x")
        reference = encode_reference([("case", [1.0])])
        with self.assertRaises(ContractError):
            decode_reference(reference[:-1])
        changed_tolerance = bytearray(encode_bundle([suites()["softmax"][0]]))
        struct.pack_into("<f", changed_tolerance, 12 + 4 + 32 + 14 * 4 + 4, 0.125)
        with self.assertRaises(ContractError):
            decode_bundle(bytes(changed_tolerance))

    def test_every_output_is_checked_and_nonfinite_fails(self):
        case = suites()["softmax"][0]
        ref = compute_reference(case)
        passed = check_output(case, ref, ref)
        self.assertEqual(passed["checked_outputs"], case.rows * case.cols)
        self.assertEqual(passed["failed_outputs"], 0)
        actual = list(ref)
        actual[0] = float("inf")
        failed = check_output(case, ref, actual)
        self.assertEqual(failed["nonfinite_outputs"], 1)
        self.assertFalse(failed["passed"])
        actual = list(ref)
        actual[0] += 0.1
        failed = check_output(case, ref, actual)
        self.assertGreater(failed["failed_outputs"], 0)
        self.assertFalse(failed["passed"])


if __name__ == "__main__":
    unittest.main()
