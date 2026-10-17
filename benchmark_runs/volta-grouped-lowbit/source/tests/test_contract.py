import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract import (
    Assignment,
    GroupedFixture,
    decode_w4a16,
    f32,
    fp16,
    gated_oracle,
    make_grouped_fixture,
    pack_signed_nibbles,
    stable_grouping,
    unpack_signed_nibbles,
    w4a16_oracle,
)


class W4A16ContractTests(unittest.TestCase):
    def test_all_sixteen_two_complement_codes_and_odd_tail(self):
        codes = list(range(-8, 8))
        packed = pack_signed_nibbles(codes)
        self.assertEqual(unpack_signed_nibbles(packed, len(codes)), codes)
        odd_codes = [-8, -1, 0, 7, 3]
        odd_packed = pack_signed_nibbles(odd_codes)
        self.assertEqual(len(odd_packed), 3)
        self.assertEqual(odd_packed[-1] >> 4, 0)
        # The unused high nibble is outside the logical tensor and ignored.
        dirty_padding = odd_packed[:-1] + bytes([odd_packed[-1] | 0xF0])
        self.assertEqual(unpack_signed_nibbles(dirty_padding, len(odd_codes)), odd_codes)

    def test_k_tail_and_wide_bounded_scale_range(self):
        k, n, group_size = 131, 3, 128
        codes = [((index * 7 + 3) % 16) - 8 for index in range(k * n)]
        packed = pack_signed_nibbles(codes)  # 393 codes exercises odd final nibble.
        scales = [2.0 ** -14, 1.0, 8000.0, 2.0 ** -13, 2.0, 4000.0]
        decoded = decode_w4a16(packed, scales, k, n, group_size)
        self.assertEqual(len(decoded), k * n)
        self.assertTrue(all(math.isfinite(value) for value in decoded))
        self.assertLessEqual(max(abs(value) for value in decoded), 65504.0)
        self.assertEqual(decode_w4a16(pack_signed_nibbles([-8]), [8000.0], 1, 1), [-64000.0])
        self.assertEqual(decode_w4a16(pack_signed_nibbles([7]), [2.0 ** -14], 1, 1),
                         [fp16(f32(7.0 * (2.0 ** -14)))])
        # Each K tail code selects the second K-axis scale group.
        for col in range(n):
            index = 128 * n + col
            expected = fp16(f32(f32(scales[n + col]) * codes[index]))
            self.assertEqual(decoded[index], expected)

    def test_scale_multiply_then_half_round_is_not_reassociated(self):
        scale = f32(0.020010920241475105)
        self.assertEqual(fp16(f32(scale * -7)), -0.14013671875)
        early_scale_round = fp16(scale)
        self.assertEqual(fp16(early_scale_round * -7), -0.1400146484375)
        self.assertNotEqual(fp16(f32(scale * -7)), fp16(early_scale_round * -7))
        self.assertEqual(decode_w4a16(pack_signed_nibbles([-7]), [scale], 1, 1, 128),
                         [-0.14013671875])

    def test_long_k_cancellation_fp64_oracle_uses_decoded_half_values(self):
        k, n, m = 257, 2, 2
        codes = [1 for _kk in range(k) for _ in range(n)]
        # Constant per-group scale leaves a one-element odd-K residual after
        # 128 pairs cancel; this checks long-K cancellation and the K tail.
        scales = [0.125] * 6
        a = []
        for row in range(m):
            for kk in range(k):
                a.append((1.0 if kk % 2 == 0 else -1.0) * (row + 1) / 8.0)
        actual = w4a16_oracle(a, m, k, n, pack_signed_nibbles(codes), scales, 128)
        self.assertEqual(actual[0], [0.015625, 0.015625])
        self.assertEqual(actual[1], [0.03125, 0.03125])

    def test_updated_weights_and_scales_are_redecoded_each_call(self):
        first = decode_w4a16(pack_signed_nibbles([1, 1]), [0.5, 0.25], 1, 2)
        second = decode_w4a16(pack_signed_nibbles([2, -2]), [1.5, 2.0], 1, 2)
        self.assertEqual(first, [0.5, 0.25])
        self.assertEqual(second, [3.0, -4.0])
        self.assertNotEqual(first, second)

    def test_invalid_packing_shapes_scales_and_finite_bounds_reject(self):
        for bad in (-9, 8, True, 1.5):
            with self.subTest(code=bad), self.assertRaises(ValueError):
                pack_signed_nibbles([bad])
        with self.assertRaises(ValueError):
            unpack_signed_nibbles(b'\x00', 3)
        with self.assertRaises(ValueError):
            decode_w4a16(pack_signed_nibbles([1]), [math.inf], 1, 1)
        with self.assertRaises(ValueError):
            decode_w4a16(pack_signed_nibbles([1]), [-1.0], 1, 1)
        with self.assertRaises(ValueError):
            decode_w4a16(pack_signed_nibbles([7]), [10000.0], 1, 1)


class GroupedGatedContractTests(unittest.TestCase):
    def test_skewed_grouping_preserves_empty_groups_duplicates_and_positions(self):
        fixture = make_grouped_fixture(skewed=True)
        offsets, grouped_indices = stable_grouping(fixture)
        counts = [offsets[i + 1] - offsets[i] for i in range(fixture.experts)]
        self.assertEqual(counts, [0, 0, 1, 0, 0, 6])
        self.assertEqual(grouped_indices, (3, 0, 1, 2, 4, 5, 6))
        self.assertEqual(sorted(grouped_indices), list(range(len(fixture.assignments))))
        duplicate_indices = [i for i, a in enumerate(fixture.assignments)
                             if (a.row_id, a.expert_id) == (3, 5)]
        self.assertEqual(len(duplicate_indices), 3)
        self.assertEqual(len({fixture.assignments[i].output_position for i in duplicate_indices}), 3)

    def test_balanced_grouping_and_duplicate_choices_are_retained(self):
        fixture = make_grouped_fixture(skewed=False)
        offsets, grouped_indices = stable_grouping(fixture)
        counts = [offsets[i + 1] - offsets[i] for i in range(fixture.experts)]
        self.assertEqual(counts, [1, 2, 0, 2, 2, 0])
        self.assertEqual(sorted(grouped_indices), list(range(len(fixture.assignments))))
        self.assertEqual(sum(counts), len(fixture.assignments))

    def test_fp64_gated_reference_scatter_uses_explicit_output_positions(self):
        fixture = make_grouped_fixture(skewed=True)
        output = gated_oracle(fixture)
        self.assertEqual(len(output), len(fixture.assignments))
        duplicate_positions = [a.output_position for a in fixture.assignments
                               if (a.row_id, a.expert_id) == (3, 5)]
        self.assertGreaterEqual(len(duplicate_positions), 2)
        first = output[duplicate_positions[0]]
        for position in duplicate_positions[1:]:
            self.assertEqual(output[position], first)
        self.assertTrue(all(math.isfinite(x) for row in output for x in row))

    def test_output_positions_must_be_a_permutation(self):
        fixture = make_grouped_fixture(skewed=True)
        assignments = list(fixture.assignments)
        assignments[-1] = Assignment(assignments[-1].row_id,
                                     assignments[-1].expert_id,
                                     assignments[0].output_position)
        malformed = GroupedFixture(fixture.rows, fixture.k, fixture.n, fixture.experts,
                                   fixture.x16, fixture.up16, fixture.gate16,
                                   tuple(assignments))
        with self.assertRaises(ValueError):
            stable_grouping(malformed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
