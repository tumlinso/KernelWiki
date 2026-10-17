"""Independent FP64 oracle for the KWV-08 gated-delta state contract.

The caller owns input normalization and representation. Values arrive as the
actual stored numbers; this oracle converts those numbers to binary64 but does
not alter the operator or the inputs.
"""
from __future__ import annotations

import numpy as np


def _check_inputs(keys, values, queries, alpha, beta, initial_state):
    k = np.asarray(keys)
    v = np.asarray(values)
    q = np.asarray(queries)
    a = np.asarray(alpha)
    b = np.asarray(beta)
    state = np.asarray(initial_state)
    if k.ndim != 2 or v.ndim != 2 or q.ndim != 2:
        raise ValueError("keys, values, and queries must be rank-2 [T, dim]")
    t, dk = k.shape
    if v.shape[0] != t or q.shape != (t, dk):
        raise ValueError("time dimensions differ or query/key dimensions differ")
    dv = v.shape[1]
    if state.shape != (dv, dk):
        raise ValueError("initial state must have shape [d_v, d_k]")
    if a.shape != (t,) or b.shape != (t,):
        raise ValueError("alpha and beta must contain one scalar per step")
    arrays = (k, v, q, a, b, state)
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("the bounded finite-input contract excludes NaN/Inf")
    return tuple(np.asarray(x, dtype=np.float64) for x in arrays)


def run_sequence(keys, values, queries, alpha, beta, initial_state,
                 start: int = 0, stop: int | None = None):
    """Run [start, stop) and return every live output and the carried state."""
    k, v, q, a, b, state = _check_inputs(
        keys, values, queries, alpha, beta, initial_state
    )
    t = k.shape[0]
    end = t if stop is None else stop
    if not (0 <= start <= end <= t):
        raise ValueError("segment must satisfy 0 <= start <= stop <= T")
    outputs = np.empty((end - start, v.shape[1]), dtype=np.float64)
    for out_index, step in enumerate(range(start, end)):
        decayed = a[step] * state
        error = v[step] - decayed @ k[step]
        state = decayed + b[step] * np.outer(error, k[step])
        # Contract intentionally observes the post-update state.
        outputs[out_index] = state @ q[step]
    return outputs, state


def run_interleaved(keys, values, queries, alpha, beta, initial_states,
                    lengths):
    """Reference time-major interleaved independent sequences with ragged T."""
    k = np.asarray(keys)
    v = np.asarray(values)
    q = np.asarray(queries)
    a = np.asarray(alpha)
    b = np.asarray(beta)
    states = np.asarray(initial_states)
    lengths = np.asarray(lengths)
    if k.ndim != 3 or v.ndim != 3 or q.ndim != 3:
        raise ValueError("interleaved inputs must be [T, B, dim]")
    t, batch, dk = k.shape
    if v.shape[:2] != (t, batch) or q.shape != (t, batch, dk):
        raise ValueError("interleaved input shapes disagree")
    dv = v.shape[2]
    if a.shape != (t, batch) or b.shape != (t, batch):
        raise ValueError("interleaved gates must have shape [T, B]")
    if states.shape != (batch, dv, dk) or lengths.shape != (batch,):
        raise ValueError("initial states or lengths have the wrong shape")
    if np.any(lengths < 0) or np.any(lengths > t):
        raise ValueError("each sequence length must lie in [0, T]")
    if not np.isfinite(states).all():
        raise ValueError("initial states must be finite")

    outputs = np.full((t, batch, dv), np.nan, dtype=np.float64)
    final_states = np.empty((batch, dv, dk), dtype=np.float64)
    for seq in range(batch):
        live = int(lengths[seq])
        seq_outputs, final_states[seq] = run_sequence(
            k[:live, seq], v[:live, seq], q[:live, seq],
            a[:live, seq], b[:live, seq], states[seq]
        )
        outputs[:live, seq] = seq_outputs
    return outputs, final_states
