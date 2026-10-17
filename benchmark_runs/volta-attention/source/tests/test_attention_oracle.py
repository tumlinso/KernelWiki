import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_oracle import Tensor4, attention_fp64, attention_online


def make(shape, seed, scale=1.0):
    rng = random.Random(seed)
    count = math.prod(shape)
    return Tensor4.from_values(shape, (rng.uniform(-scale, scale) for _ in range(count)))


def assert_close_vectors(test, left, right, atol=2e-12, rtol=2e-12):
    test.assertEqual(len(left), len(right))
    for i, (a, b) in enumerate(zip(left, right)):
        test.assertLessEqual(abs(a - b), atol + rtol * abs(a), f"index {i}: {a} != {b}")


class AttentionOracleTests(unittest.TestCase):
    def test_fp16_representations_online_merges_and_block_boundaries(self):
        q = make((1, 4, 3, 7), 1, 2)
        k = make((1, 2, 9, 7), 2, 2)
        v = make((1, 2, 9, 7), 3, 2)
        ref = attention_fp64(q, k, v)
        for ranges in ([(0, 9)], [(0, 2), (2, 5), (5, 9)],
                       [(0, 1), (1, 4), (4, 6), (6, 9)]):
            actual = attention_online(q, k, v, key_ranges=ranges)
            assert_close_vectors(self, actual["output"], ref["output"])
            for a, b in zip(actual["rows"], ref["rows"]):
                self.assertAlmostEqual(a["logsumexp"], b["logsumexp"], places=12)
                self.assertAlmostEqual(a["max_score"], b["max_score"], places=12)
                self.assertAlmostEqual(a["normalizer"], b["normalizer"], places=12)
                self.assertEqual(a["valid_keys"], b["valid_keys"])

    def test_suffix_aligned_causal_mask_with_unequal_lengths(self):
        q = Tensor4.from_values((1, 1, 2, 1), [1, 1])
        k = Tensor4.from_values((1, 1, 4, 1), [0, 0, 0, 0])
        v = Tensor4.from_values((1, 1, 4, 1), [1, 2, 4, 8])
        result = attention_fp64(q, k, v, causal=True)
        rows = result["rows"]
        self.assertEqual([row["valid_keys"] for row in rows], [3, 4])
        self.assertAlmostEqual(result["output"][0], 7.0 / 3.0)
        self.assertAlmostEqual(result["output"][1], 15.0 / 4.0)
        with self.assertRaisesRegex(ValueError, "query length <= key length"):
            attention_fp64(Tensor4.from_values((1, 1, 5, 1), [1] * 5), k, v, causal=True)

    def test_all_masked_and_empty_selected_row_are_defined_zero(self):
        q = make((1, 2, 2, 3), 4)
        k = make((1, 1, 5, 3), 5)
        v = make((1, 1, 5, 3), 6)
        mask = [[[[True] * 5 for _ in range(2)] for _ in range(2)]]
        masked = attention_fp64(q, k, v, masked=mask)
        self.assertEqual(masked["output"], (0.0,) * 12)
        self.assertTrue(all(row["normalizer"] == 0.0 for row in masked["rows"]))
        selected = [[ [[], [0]], [[], [1]] ]]
        sparse = attention_fp64(q, k, v, selected_blocks=selected, block_size=3)
        for row, values in zip(sparse["rows"], (sparse["output"][0:3], sparse["output"][3:6],
                                                  sparse["output"][6:9], sparse["output"][9:12])):
            if row["valid_keys"] == 0:
                self.assertEqual(values, (0.0, 0.0, 0.0))

    def test_gqa_keeps_each_query_head_normalizer_independent(self):
        q = Tensor4.from_values((1, 4, 1, 1), [0, 1, 2, 3])
        k = Tensor4.from_values((1, 1, 3, 1), [0, 1, -1])
        v = Tensor4.from_values((1, 1, 3, 1), [2, 4, 8])
        result = attention_fp64(q, k, v)
        lses = [row["logsumexp"] for row in result["rows"]]
        outs = list(result["output"])
        self.assertEqual(len(set(lses)), 4)
        self.assertEqual(len(set(outs)), 4)
        order = [2, 0, 3, 1]
        q_reordered = Tensor4.from_values((1, 4, 1, 1), [q.values[i] for i in order])
        reordered = attention_fp64(q_reordered, k, v)
        self.assertEqual(list(reordered["output"]), [outs[i] for i in order])
        self.assertEqual([row["logsumexp"] for row in reordered["rows"]], [lses[i] for i in order])

    def test_selected_blocks_match_dense_masked_reference_and_order_is_irrelevant(self):
        q = make((1, 2, 3, 5), 7)
        k = make((1, 1, 11, 5), 8)
        v = make((1, 1, 11, 5), 9)
        # Three blocks; last block contains keys 8,9,10. Each query has an
        # independently supplied membership list, including an empty row.
        selected = [[[ [2, 0], [], [1] ], [ [1], [2, 0], [0] ]]]
        sparse = attention_fp64(q, k, v, causal=True,
                                selected_blocks=selected, block_size=4)
        dense_mask = [[[[((ki // 4) not in set(ids)) for ki in range(11)]
                        for ids in head] for head in batch] for batch in selected]
        dense = attention_fp64(q, k, v, causal=True, masked=dense_mask)
        assert_close_vectors(self, sparse["output"], dense["output"])
        reordered = [[[list(reversed(ids)) for ids in head] for head in batch]
                     for batch in selected]
        actual = attention_fp64(q, k, v, causal=True,
                                selected_blocks=reordered, block_size=4)
        assert_close_vectors(self, actual["output"], sparse["output"])
        self.assertEqual([row["valid_keys"] for row in sparse["rows"]],
                         [row["valid_keys"] for row in dense["rows"]])

    def test_duplicate_and_out_of_range_block_ids_are_rejected(self):
        q = make((1, 1, 1, 2), 10)
        k = make((1, 1, 5, 2), 11)
        v = make((1, 1, 5, 2), 12)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            attention_fp64(q, k, v, selected_blocks=[[[[1, 1]]]], block_size=2)
        with self.assertRaisesRegex(ValueError, "outside"):
            attention_fp64(q, k, v, selected_blocks=[[[[3]]]], block_size=2)

    def test_extreme_and_nearly_tied_scores_remain_finite(self):
        q = Tensor4.from_values((1, 1, 2, 2), [100, -100, 0.001, 0.001])
        k = Tensor4.from_values((1, 1, 3, 2), [100, 0, 100, 0.000001, -100, 0])
        v = Tensor4.from_values((1, 1, 3, 2), [1, -3, 2, 4, 8, 2])
        ref = attention_fp64(q, k, v)
        actual = attention_online(q, k, v, key_ranges=[(0, 1), (1, 3)])
        for value in (*ref["output"], *actual["output"]):
            self.assertTrue(math.isfinite(value))
        assert_close_vectors(self, actual["output"], ref["output"], atol=1e-10, rtol=1e-10)

    def test_input_changes_do_not_reuse_any_prior_row_state(self):
        q1, k1, v1 = make((1, 1, 1, 4), 20), make((1, 1, 8, 4), 21), make((1, 1, 8, 4), 22)
        q2, k2, v2 = make((1, 1, 1, 4), 23, 8), make((1, 1, 8, 4), 24, 8), make((1, 1, 8, 4), 25, 8)
        first = attention_fp64(q1, k1, v1)
        second = attention_fp64(q2, k2, v2)
        again = attention_fp64(q1, k1, v1)
        self.assertNotEqual(first["output"], second["output"])
        self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()
