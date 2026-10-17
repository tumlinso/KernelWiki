#!/usr/bin/env python3
"""Small CPU/NumPy sanity checks of the design equations, not GPU qualification."""
from __future__ import annotations
import json
import sys
import numpy as np


def state(scores: np.ndarray, values: np.ndarray):
    keep = np.isfinite(scores)
    if not np.any(keep):
        return -np.inf, 0.0, np.zeros(values.shape[-1])
    m = float(np.max(scores[keep]))
    weights = np.exp(scores[keep] - m)
    return m, float(np.sum(weights)), weights @ values[keep]


def merge(a, b):
    if a[1] == 0:
        return b
    if b[1] == 0:
        return a
    m = max(a[0], b[0])
    wa, wb = np.exp(a[0] - m), np.exp(b[0] - m)
    return m, wa * a[1] + wb * b[1], wa * a[2] + wb * b[2]


def finish(s):
    return s[2] / s[1] if s[1] else np.zeros_like(s[2])


def update(s, k, v, alpha, beta):
    d = alpha * s
    return d + beta * np.outer(v - d @ k, k)


def main():
    rng = np.random.default_rng(701710)
    groups = []
    # Independent reference materializes one normalized score vector.
    count, maxerr = 0, 0.0
    for n in (1, 7, 32, 65):
        v = rng.normal(size=(n, 5))
        scores = 10 * rng.normal(size=n)
        for all_masked in (False, True):
            x = scores.copy()
            x[::3] = -np.inf
            if all_masked:
                x[:] = -np.inf
            good = np.isfinite(x)
            expected = np.zeros(5)
            if np.any(good):
                w = np.exp(x[good] - np.max(x[good]))
                expected = (w / np.sum(w)) @ v[good]
            pieces = [state(x[ix], v[ix]) for ix in np.array_split(np.arange(n), 4)]
            for ordered in (pieces, list(reversed(pieces))):
                s = (-np.inf, 0.0, np.zeros(5))
                for piece in ordered:
                    s = merge(s, piece)
                out = finish(s)
                np.testing.assert_allclose(out, expected, rtol=2e-12, atol=2e-12)
                maxerr = max(maxerr, float(np.max(np.abs(out - expected))))
                count += 1
    groups.append({'name': 'online_softmax_merge_empty_masks_partition_order', 'cases': count, 'max_abs_error': maxerr})

    maxerr = 0.0
    for alpha, beta in ((0.,0.), (0.,1.), (1.,0.), (1.,1.), (.97,.4)):
        s, k, v = rng.normal(size=(3,4)), rng.normal(size=4), rng.normal(size=3)
        direct = update(s, k, v, alpha, beta)
        expanded = alpha * s @ (np.eye(4) - beta * np.outer(k,k)) + beta * np.outer(v,k)
        np.testing.assert_allclose(direct, expanded, rtol=2e-12, atol=2e-12)
        maxerr = max(maxerr, float(np.max(np.abs(direct-expanded))))
    groups.append({'name': 'gated_delta_orientation_and_gate_placement', 'cases': 5, 'max_abs_error': maxerr})

    keys = rng.normal(size=(37,4)); keys /= np.linalg.norm(keys,axis=1)[:,None]
    vals = rng.normal(size=(37,3)); alphas = rng.uniform(.9,1,size=37); betas = rng.uniform(0,1,size=37)
    def roll(start, lo, hi):
        for i in range(lo,hi):
            start = update(start, keys[i], vals[i], alphas[i], betas[i])
        return start
    initial = rng.normal(size=(3,4))
    whole = roll(initial.copy(),0,37)
    for split in (0,1,17,36,37):
        partitioned = roll(roll(initial.copy(),0,split),split,37)
        np.testing.assert_array_equal(whole, partitioned)
    groups.append({'name':'recurrent_state_carry_preserves_full_history','cases':5,'max_abs_error':0.0})

    for i in range(8):
        k = rng.normal(size=4); k /= np.linalg.norm(k)
        alpha, beta = rng.uniform(0,1,size=2)
        factor = alpha * (np.eye(4) - beta*np.outer(k,k))
        assert np.linalg.norm(factor,ord=2) <= alpha + 1e-12
    groups.append({'name':'unit_key_inherited_error_contraction_sanity','cases':8,
                   'limit':'Finite numerical illustration; analytical bound is derived in M05, not established by sampling.'})

    q, k, v = rng.normal(size=(3,8)), rng.normal(size=(11,8)), rng.normal(size=(11,5))
    masks = rng.random(size=(3,11)) > .25
    packed_scores = q @ k.T / np.sqrt(8)
    out = np.stack([finish(state(np.where(masks[i], packed_scores[i], -np.inf), v)) for i in range(3)])
    independent = np.stack([finish(state(np.where(masks[i], k @ q[i]/np.sqrt(8), -np.inf),v)) for i in range(3)])
    np.testing.assert_allclose(out,independent,rtol=2e-12,atol=2e-12)
    groups.append({'name':'gqa_rows_share_kv_not_normalizers','cases':3,'max_abs_error':float(np.max(np.abs(out-independent)))})

    for lo in range(16):
        for hi in range(16):
            packed = lo | (hi << 4)
            decoded = [(packed >> shift) & 15 for shift in (0,4)]
            signed = [x-16 if x>=8 else x for x in decoded]
            assert signed == [lo-16 if lo>=8 else lo, hi-16 if hi>=8 else hi]
    groups.append({'name':'signed_int4_two_nibbles_all_encodings','cases':256})
    print(json.dumps({'status':'passed','python':sys.version.split()[0],'numpy':np.__version__,
                      'groups':groups,'limits':['CPU float64/reference algebra only; no CUDA kernel or Tensor Core behavior tested.',
                      'These tolerances are not acceptance tolerances for a future mixed-precision implementation.']}, indent=2))


if __name__ == '__main__':
    main()
