#!/usr/bin/env python3
"""Prepare the immutable FP32 input + independent FP64 reference bundle."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bundle_io import BundleCase, write_bundle
from recurrence_oracle import run_interleaved


def _lengths(t: int, batch: int, rng: np.random.Generator) -> np.ndarray:
    available = np.array(sorted(set([0, 1, min(8, t), min(32, t),
                                     min(33, t), min(127, t),
                                     max(0, t - 1), t])), dtype=np.int32)
    lengths = rng.choice(available, size=batch).astype(np.int32)
    fixed = [0, 1, min(8, t), min(33, t), t, t, min(127, t), max(0, t - 1)]
    for i, length in enumerate(fixed[:batch]):
        lengths[i] = length
    return lengths


def make_case(case_id: str, t: int, batch: int, dv: int, dk: int,
              seed: int) -> BundleCase:
    rng = np.random.default_rng(seed)
    lengths = _lengths(t, batch, rng)
    keys = rng.normal(size=(t, batch, dk)).astype(np.float32)
    norm = np.linalg.norm(keys.astype(np.float64), axis=2, keepdims=True)
    keys = (keys.astype(np.float64) / norm).astype(np.float32)
    values = (0.05 * rng.normal(size=(t, batch, dv))).astype(np.float32)
    queries = (0.2 * rng.normal(size=(t, batch, dk))).astype(np.float32)
    alpha = rng.uniform(0.80, 1.0, size=(t, batch)).astype(np.float32)
    beta = rng.uniform(0.05, 0.95, size=(t, batch)).astype(np.float32)
    initial = (0.025 * rng.normal(size=(batch, dv, dk))).astype(np.float32)

    # The short sequence exercises repeated then orthogonal unit keys and
    # mixed-sign values without adding another native shape or receipt set.
    if batch > 2 and dk > 1 and lengths[2] >= 8:
        keys[:3, 2, :] = 0.0
        keys[:3, 2, 0] = 1.0
        keys[3:6, 2, :] = 0.0
        keys[3:6, 2, 1] = 1.0
        keys[6:8, 2, :] = 0.0
        keys[6:8, 2, 0] = -1.0
        values[:8, 2] = np.where(
            (np.arange(8)[:, None] + np.arange(dv)[None, :]) % 2 == 0,
            np.float32(0.05), np.float32(-0.05)
        )

    active_alpha = np.float32(1.0 - 2.0**-12)
    active_beta = np.float32(1.0 - 2.0**-10)
    # A full long sequence isolates the near-one gate stress. A second long
    # sequence receives boundary gates inside history, rather than only at t=0.
    if t > 1 and batch > 4 and lengths[4] == t:
        alpha[:, 4] = active_alpha
        beta[:, 4] = active_beta
        if t > 93:
            alpha[93, 4] = np.float32(0.0)
        if t > 94:
            beta[94, 4] = np.float32(0.0)
    if t > 1 and batch > 5 and lengths[5] == t:
        alpha[:, 5] = active_alpha
        beta[:, 5] = active_beta
    if t > 0 and batch > 1 and lengths[1] > 0:
        alpha[0, 1] = np.float32(0.0)
        beta[0, 1] = np.float32(0.0)
    if t > 0 and batch > 2 and lengths[2] > 0:
        beta[0, 2] = np.float32(0.0)
    if t > 0 and batch > 3 and lengths[3] > 0:
        alpha[0, 3] = np.float32(0.0)

    # NaN marks storage padding. The native kernel must leave its output padding
    # sentinel untouched and must preserve zero-length initial states.
    for seq, length in enumerate(lengths):
        if length < t:
            keys[length:, seq] = np.nan
            values[length:, seq] = np.nan
            queries[length:, seq] = np.nan
            alpha[length:, seq] = np.nan
            beta[length:, seq] = np.nan

    outputs, final_states = run_interleaved(
        keys, values, queries, alpha, beta, initial, lengths
    )
    live = np.arange(t)[:, None] < lengths[None, :]
    if not np.isfinite(outputs[live]).all() or not np.isfinite(final_states).all():
        raise RuntimeError("FP64 oracle produced a non-finite live result")
    if not np.isfinite(initial).all():
        raise RuntimeError("generated initial state is not finite")
    return BundleCase(case_id, keys, values, queries, alpha, beta,
                      lengths, initial, outputs, final_states)


def build_bundle():
    return [
        make_case("decode_many_sequences", 1, 256, 32, 32, 0x080801),
        make_case("short_prefill_many_sequences", 33, 128, 32, 32, 0x080802),
        make_case("long_prefill_many_sequences", 257, 128, 64, 64, 0x080803),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases = build_bundle()
    manifest = write_bundle(args.output, cases)
    print(f"wrote {args.output} ({manifest['bundle_bytes']} bytes)")
    print(f"bundle SHA-256: {manifest['bundle_file_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
