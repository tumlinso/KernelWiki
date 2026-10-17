import unittest

import numpy as np

from recurrence_oracle import run_interleaved, run_sequence


class RecurrenceOracleTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(81008)

    def sequence(self, t=9, dv=3, dk=4):
        # Generate represented FP32 inputs, then let the independent reference
        # consume those exact stored values in FP64.
        keys = self.rng.normal(size=(t, dk)).astype(np.float32)
        values = self.rng.normal(size=(t, dv)).astype(np.float32)
        queries = self.rng.normal(size=(t, dk)).astype(np.float32)
        alpha = self.rng.uniform(0.0, 1.0, size=t).astype(np.float32)
        beta = self.rng.uniform(0.0, 1.0, size=t).astype(np.float32)
        state = self.rng.normal(size=(dv, dk)).astype(np.float32)
        return keys, values, queries, alpha, beta, state

    def test_zero_and_nonzero_initial_state_and_boundary_gates(self):
        args = list(self.sequence(t=5))
        args[3] = np.array([1.0, 1.0, 0.0, 0.75, 1.0], np.float32)
        args[4] = np.array([0.0, 0.0, 1.0, 0.0, 1.0], np.float32)
        zero = np.zeros_like(args[5])
        out0, final0 = run_sequence(*args[:5], zero)
        outn, finaln = run_sequence(*args)
        self.assertTrue(np.isfinite(out0).all())
        self.assertTrue(np.isfinite(final0).all())
        self.assertTrue(np.isfinite(outn).all())
        self.assertTrue(np.isfinite(finaln).all())
        self.assertGreater(float(np.max(np.abs(outn[1] - out0[1]))), 1e-5)
        # alpha=0 at step 2 removes inherited state exactly before the update.
        _, state_at_2 = run_sequence(*args[:5], args[5], stop=3)
        expected = float(args[4][2]) * np.outer(
            args[1][2].astype(np.float64), args[0][2].astype(np.float64)
        )
        np.testing.assert_allclose(state_at_2, expected, rtol=0, atol=2e-15)

    def test_beta_zero_is_decay_only(self):
        keys, values, queries, alpha, _, state = self.sequence(t=6)
        beta = np.zeros_like(alpha)
        outputs, final = run_sequence(keys, values, queries, alpha, beta, state)
        expected_state = state.astype(np.float64)
        expected_outputs = []
        for i in range(len(alpha)):
            expected_state *= float(alpha[i])
            expected_outputs.append(expected_state @ queries[i].astype(np.float64))
        np.testing.assert_allclose(final, expected_state, rtol=0, atol=2e-15)
        np.testing.assert_allclose(outputs, expected_outputs, rtol=0, atol=2e-15)

    def test_post_update_output_and_mixed_sign_repeated_orthogonal_keys(self):
        keys = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
        values = np.array([[1, -2], [-3, 4], [5, -6], [-7, 8]], dtype=np.float32)
        queries = np.array([[1, 2], [-1, 3], [4, -2], [3, 1]], dtype=np.float32)
        alpha = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        beta = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        initial = np.array([[2, -1], [0.5, 3]], dtype=np.float32)
        outputs, final = run_sequence(keys, values, queries, alpha, beta, initial)
        state = initial.astype(np.float64)
        manual = []
        for i in range(len(alpha)):
            decayed = float(alpha[i]) * state
            error = values[i].astype(np.float64) - decayed @ keys[i]
            state = decayed + float(beta[i]) * np.outer(error, keys[i])
            manual.append(state @ queries[i])
        np.testing.assert_allclose(outputs, manual, rtol=0, atol=0)
        np.testing.assert_allclose(final, state, rtol=0, atol=0)

    def test_expanded_rank_one_identity_matches_each_recurrence_state(self):
        keys, values, queries, alpha, beta, initial = self.sequence(t=11, dv=4, dk=6)
        outputs, final = run_sequence(keys, values, queries, alpha, beta, initial)
        state = initial.astype(np.float64)
        identity = np.eye(keys.shape[1], dtype=np.float64)
        expanded_outputs = []
        for i in range(len(alpha)):
            k = keys[i].astype(np.float64)
            v = values[i].astype(np.float64)
            state = (
                float(alpha[i]) * state
                @ (identity - float(beta[i]) * np.outer(k, k))
                + float(beta[i]) * np.outer(v, k)
            )
            expanded_outputs.append(state @ queries[i].astype(np.float64))
        np.testing.assert_allclose(outputs, expanded_outputs, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(final, state, rtol=2e-14, atol=2e-14)

    def test_segmented_execution_carries_returned_state_without_reset(self):
        keys, values, queries, alpha, beta, initial = self.sequence(t=17, dv=5, dk=7)
        whole_outputs, whole_final = run_sequence(
            keys, values, queries, alpha, beta, initial
        )
        cuts = [0, 1, 6, 6, 12, 17]
        carried = initial.astype(np.float64)
        segmented_outputs = []
        for lo, hi in zip(cuts, cuts[1:]):
            out, carried = run_sequence(
                keys, values, queries, alpha, beta, carried, start=lo, stop=hi
            )
            segmented_outputs.append(out)
        np.testing.assert_array_equal(np.concatenate(segmented_outputs), whole_outputs)
        np.testing.assert_array_equal(carried, whole_final)

    def test_long_near_one_gates_keeps_every_output_and_final_state(self):
        t, dv, dk = 1200, 3, 4
        keys = self.rng.normal(size=(t, dk)).astype(np.float32)
        keys /= np.linalg.norm(keys.astype(np.float64), axis=1)[:, None].astype(np.float32)
        values = (self.rng.normal(size=(t, dv)) * 0.1).astype(np.float32)
        queries = self.rng.normal(size=(t, dk)).astype(np.float32)
        alpha = np.full(t, 1.0 - 2.0**-12, dtype=np.float32)
        beta = np.full(t, 1.0 - 2.0**-10, dtype=np.float32)
        initial = (self.rng.normal(size=(dv, dk)) * 0.1).astype(np.float32)
        outputs, final = run_sequence(keys, values, queries, alpha, beta, initial)
        self.assertEqual(outputs.shape, (t, dv))
        self.assertEqual(final.shape, (dv, dk))
        self.assertTrue(np.isfinite(outputs).all())
        self.assertTrue(np.isfinite(final).all())
        # The example bound is relevant only because these test keys are unit
        # length and both gate streams lie in [0, 1]. It does not set the GPU
        # tolerance or claim a measured hardware error bound.
        self.assertLess(np.max(np.abs(np.linalg.norm(keys.astype(np.float64), axis=1) - 1)), 2e-7)
        self.assertTrue(np.all((0 <= alpha) & (alpha <= 1)))
        self.assertTrue(np.all((0 <= beta) & (beta <= 1)))

    def test_time_major_ragged_sequences_do_not_mix_or_update_padding(self):
        t, batch, dv, dk = 13, 5, 3, 4
        keys = self.rng.normal(size=(t, batch, dk)).astype(np.float32)
        values = self.rng.normal(size=(t, batch, dv)).astype(np.float32)
        queries = self.rng.normal(size=(t, batch, dk)).astype(np.float32)
        alpha = self.rng.uniform(0.0, 1.0, size=(t, batch)).astype(np.float32)
        beta = self.rng.uniform(0.0, 1.0, size=(t, batch)).astype(np.float32)
        lengths = np.array([13, 1, 0, 8, 5], dtype=np.int32)
        states = self.rng.normal(size=(batch, dv, dk)).astype(np.float32)
        outputs, finals = run_interleaved(
            keys, values, queries, alpha, beta, states, lengths
        )
        for seq, length in enumerate(lengths):
            self.assertTrue(np.isnan(outputs[length:, seq]).all())
            ref_out, ref_final = run_sequence(
                keys[:length, seq], values[:length, seq], queries[:length, seq],
                alpha[:length, seq], beta[:length, seq], states[seq]
            )
            np.testing.assert_array_equal(outputs[:length, seq], ref_out)
            np.testing.assert_array_equal(finals[seq], ref_final)

        # A bijective batch permutation changes interleaving but cannot change
        # any sequence's state history or output.
        order = np.array([3, 0, 4, 2, 1])
        permuted = run_interleaved(
            keys[:, order], values[:, order], queries[:, order],
            alpha[:, order], beta[:, order], states[order], lengths[order]
        )
        np.testing.assert_array_equal(permuted[0][:, np.argsort(order)], outputs)
        np.testing.assert_array_equal(permuted[1][np.argsort(order)], finals)

    def test_sequence_order_is_semantic_and_cannot_be_used_as_a_reset(self):
        args = self.sequence(t=7, dv=2, dk=3)
        out, final = run_sequence(*args)
        rev_args = [x[::-1].copy() for x in args[:5]] + [args[5]]
        rev_out, rev_final = run_sequence(*rev_args)
        self.assertGreater(float(np.max(np.abs(final - rev_final))), 1e-5)
        self.assertGreater(float(np.max(np.abs(out - rev_out[::-1]))), 1e-5)

    def test_invalid_segment_and_unbounded_nonfinite_inputs_fail_closed(self):
        args = self.sequence(t=4)
        with self.assertRaises(ValueError):
            run_sequence(*args, start=3, stop=2)
        bad = list(args)
        bad[0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            run_sequence(*bad)


if __name__ == "__main__":
    unittest.main()
