import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_bundle import (INPUT_HEADER, INPUT_RECORD, InputCase, ReferenceCase,
                             read_input_bundle, read_reference_bundle,
                             write_input_bundle, write_reference_bundle)
from attention_oracle import Tensor4, attention_fp64


class AttentionBundleTests(unittest.TestCase):
    def make_case(self):
        q = Tensor4.from_values((1, 2, 2, 3), [1, 0, 0, 0, 1, 0,
                                              0.5, 0.5, 0, -0.5, 0.5, 0])
        k = Tensor4.from_values((1, 1, 3, 3), [1, 0, 0, 0, 1, 0, -1, 0, 0])
        v = Tensor4.from_values((1, 1, 3, 3), [2, 3, 4, 5, 6, 7, 8, 9, 10])
        selected = [[[[0], [1]], [[1], []]]]
        mask = [[[[False, True, False], [False, False, False]],
                 [[False, False, True], [False, False, False]]]]
        case = InputCase("bundle-test", q, k, v, False, 2, selected, mask)
        expected = attention_fp64(q, k, v, selected_blocks=selected,
                                  block_size=2, masked=mask)
        reference = ReferenceCase(case.name, expected["shape"], k.shape,
                                  tuple(expected["output"]), tuple(expected["rows"]))
        return case, reference

    def test_input_and_fp64_reference_binary_round_trip(self):
        case, reference = self.make_case()
        with tempfile.TemporaryDirectory() as temp:
            input_path = Path(temp) / "inputs.bin"
            reference_path = Path(temp) / "reference.bin"
            records = write_input_bundle(input_path, [case])
            write_reference_bundle(reference_path, [reference])
            parsed = read_input_bundle(input_path)
            self.assertEqual(records[0]["id"], case.name)
            self.assertEqual(len(parsed), 1)
            actual = parsed[0]
            self.assertEqual(actual["q"], case.q)
            self.assertEqual(actual["k"], case.k)
            self.assertEqual(actual["v"], case.v)
            self.assertEqual(actual["selected_blocks"], [[[[0], [1]], [[1], []]]])
            self.assertEqual(actual["masked"], [[[[False, True, False],
                                                   [False, False, False]],
                                                  [[False, False, True],
                                                   [False, False, False]]]])
            expected = read_reference_bundle(reference_path, parsed)[case.name]
            self.assertEqual(expected.output, reference.output)
            recomputed = attention_fp64(actual["q"], actual["k"], actual["v"],
                                        selected_blocks=actual["selected_blocks"],
                                        block_size=actual["block_size"],
                                        masked=actual["masked"])
            self.assertEqual(recomputed["output"], expected.output)
            for actual_row, reference_row in zip(expected.rows, reference.rows):
                for key in ("max_score", "normalizer", "logsumexp", "valid_keys"):
                    self.assertEqual(actual_row[key], reference_row[key])

    def test_input_reader_rejects_out_of_bounds_section(self):
        case, _ = self.make_case()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inputs.bin"
            write_input_bundle(path, [case])
            raw = bytearray(path.read_bytes())
            offset = INPUT_HEADER.size
            fields = list(INPUT_RECORD.unpack_from(raw, offset))
            fields[11] = len(raw) + 32  # first table uint64 is Q byte offset
            INPUT_RECORD.pack_into(raw, offset, *fields)
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "Q exceeds"):
                read_input_bundle(path)

    def test_reference_reader_rejects_wrong_case_order(self):
        case, reference = self.make_case()
        with tempfile.TemporaryDirectory() as temp:
            input_path = Path(temp) / "inputs.bin"
            reference_path = Path(temp) / "reference.bin"
            write_input_bundle(input_path, [case])
            write_reference_bundle(reference_path,
                                   [ReferenceCase("wrong-case", reference.shape,
                                                  reference.kv_shape, reference.output,
                                                  reference.rows)])
            parsed = read_input_bundle(input_path)
            with self.assertRaisesRegex(ValueError, "reference metadata differs"):
                read_reference_bundle(reference_path, parsed)


if __name__ == "__main__":
    unittest.main()
