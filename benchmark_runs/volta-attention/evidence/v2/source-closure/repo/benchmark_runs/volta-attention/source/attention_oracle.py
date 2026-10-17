#!/usr/bin/env python3
"""Pure-Python FP64 oracle and stable block-partitioned attention reference.

No NumPy, CUDA, or repository imports are required. Tensor storage is row-major
[B, H, S, D]. Inputs should be quantized through Tensor4.from_values to model
the actual FP16 representations supplied to a native implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Tensor4:
    shape: tuple[int, int, int, int]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != 4 or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ValueError("tensor shape must contain four positive integers")
        count = math.prod(self.shape)
        if len(self.values) != count:
            raise ValueError(f"tensor has {len(self.values)} values, expected {count}")
        if not all(math.isfinite(x) for x in self.values):
            raise ValueError("the initial attention contract accepts finite inputs only")

    @classmethod
    def from_values(cls, shape: Sequence[int], values: Iterable[float]) -> "Tensor4":
        # struct's binary16 conversion is part of the fixture, so oracle inputs
        # exactly match values representable in the native FP16 input buffer.
        quantized = tuple(struct.unpack("<e", struct.pack("<e", float(v)))[0]
                          for v in values)
        return cls(tuple(shape), quantized)  # type: ignore[arg-type]

    def at(self, b: int, h: int, s: int, d: int) -> float:
        _, heads, rows, width = self.shape
        return self.values[(((b * heads + h) * rows + s) * width) + d]


def _validate(q: Tensor4, k: Tensor4, v: Tensor4, causal: bool) -> tuple[int, ...]:
    bq, hq, sq, d = q.shape
    bk, hk, sk, dk = k.shape
    bv, hv, sv, dv = v.shape
    if bq != bk or bq != bv or hk != hv or sk != sv or d != dk or d != dv:
        raise ValueError("Q/K/V batch, KV heads, sequence, and feature dimensions disagree")
    if hq % hk:
        raise ValueError("query head count must be divisible by KV head count")
    if causal and sq > sk:
        raise ValueError("suffix-aligned causal attention requires query length <= key length")
    return bq, hq, hk, sq, sk, d


def _validate_row_mask(mask: Sequence[Sequence[Sequence[Sequence[bool]]]] | None,
                       b: int, hq: int, sq: int, sk: int) -> None:
    if mask is None:
        return
    if len(mask) != b or any(len(batch) != hq for batch in mask):
        raise ValueError("mask batch/head shape differs from Q")
    for batch in mask:
        for head in batch:
            if len(head) != sq or any(len(row) != sk for row in head):
                raise ValueError("mask query/key shape differs from Q/K")
            for row in head:
                if any(type(flag) is not bool for flag in row):
                    raise ValueError("mask entries must be booleans (True means masked)")


def _validate_selection_shape(
        selected: Sequence[Sequence[Sequence[Sequence[int]]]] | None,
        b: int, hq: int, sq: int) -> None:
    if selected is None:
        return
    if len(selected) != b or any(len(batch) != hq for batch in selected):
        raise ValueError("selected-block batch/head shape differs from Q")
    if any(len(head) != sq for batch in selected for head in batch):
        raise ValueError("selected-block query shape differs from Q")


def _selected_for_row(selected_blocks: Sequence[Sequence[Sequence[Sequence[int]]]] | None,
                      b: int, h: int, qrow: int, block_count: int) -> set[int] | None:
    if selected_blocks is None:
        return None
    try:
        ids = list(selected_blocks[b][h][qrow])
    except (IndexError, TypeError) as exc:
        raise ValueError("selected-block index shape differs from Q") from exc
    if any(type(i) is not int for i in ids):
        raise ValueError("selected block IDs must be integers")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate selected block ID for query row {(b, h, qrow)}")
    if any(i < 0 or i >= block_count for i in ids):
        raise ValueError(f"selected block ID outside [0, {block_count})")
    return set(ids)


def _allowed(b: int, h: int, qi: int, ki: int, *, sk: int, sq: int,
             causal: bool,
             mask: Sequence[Sequence[Sequence[Sequence[bool]]]] | None,
             selected: set[int] | None, block_size: int) -> bool:
    if causal and ki > sk - sq + qi:
        return False
    if mask is not None and mask[b][h][qi][ki]:
        return False
    if selected is not None and ki // block_size not in selected:
        return False
    return True


def _score(q: Tensor4, k: Tensor4, b: int, hq: int, hk: int,
           qi: int, ki: int, d: int) -> float:
    # Python float arithmetic is binary64 on the supported host runtimes.
    dot = math.fsum(q.at(b, hq, qi, x) * k.at(b, hk, ki, x) for x in range(d))
    return dot / math.sqrt(d)


def _row_keys(q: Tensor4, k: Tensor4, b: int, h: int, qi: int,
              *, causal: bool,
              mask: Sequence[Sequence[Sequence[Sequence[bool]]]] | None,
              selected: set[int] | None, block_size: int) -> list[tuple[int, float]]:
    hk = h // (q.shape[1] // k.shape[1])
    sk = k.shape[2]
    return [(ki, _score(q, k, b, h, hk, qi, ki, q.shape[3]))
            for ki in range(sk)
            if _allowed(b, h, qi, ki, sk=sk, sq=q.shape[2], causal=causal,
                        mask=mask, selected=selected, block_size=block_size)]


def _reference_state(scores_and_ids: Sequence[tuple[int, float]], v: Tensor4,
                     b: int, hk: int, dv: int) -> tuple[list[float], float, float, float]:
    if not scores_and_ids:
        return [0.0] * dv, -math.inf, 0.0, -math.inf
    m = max(score for _, score in scores_and_ids)
    weights = [(ki, math.exp(score - m)) for ki, score in scores_and_ids]
    l = math.fsum(weight for _, weight in weights)
    u = [math.fsum(weight * v.at(b, hk, ki, d) for ki, weight in weights)
         for d in range(dv)]
    return [x / l for x in u], m, l, m + math.log(l)


def attention_fp64(q: Tensor4, k: Tensor4, v: Tensor4, *, causal: bool = False,
                   masked: Sequence[Sequence[Sequence[Sequence[bool]]]] | None = None,
                   selected_blocks: Sequence[Sequence[Sequence[Sequence[int]]]] | None = None,
                   block_size: int = 1) -> dict[str, object]:
    """Materialized-score semantic oracle. True mask entries are excluded.

    Causal query positions are right/suffix aligned: query row i is at key
    position Sk-Sq+i. Every GQA query head gets its own score normalization.
    Duplicate selected block IDs are rejected; empty selections return zero.
    """
    bsz, hq, hkv, sq, sk, d = _validate(q, k, v, causal)
    _validate_row_mask(masked, bsz, hq, sq, sk)
    _validate_selection_shape(selected_blocks, bsz, hq, sq)
    if selected_blocks is not None and (type(block_size) is not int or block_size <= 0):
        raise ValueError("selected-block attention requires a positive integer block size")
    nblocks = math.ceil(sk / block_size) if selected_blocks is not None else 0
    output: list[float] = []
    row_stats: list[dict[str, float | int]] = []
    for b in range(bsz):
        for h in range(hq):
            hk = h // (hq // hkv)
            for qi in range(sq):
                selected = _selected_for_row(selected_blocks, b, h, qi, nblocks)
                entries = _row_keys(q, k, b, h, qi, causal=causal, mask=masked,
                                    selected=selected, block_size=block_size)
                row_output, m, l, lse = _reference_state(entries, v, b, hk, d)
                output.extend(row_output)
                row_stats.append({"batch": b, "query_head": h, "query_row": qi,
                                  "max_score": m, "normalizer": l, "logsumexp": lse,
                                  "valid_keys": len(entries)})
    return {"shape": (bsz, hq, sq, d), "output": tuple(output), "rows": row_stats}


def _partial_state(entries: Sequence[tuple[int, float]], v: Tensor4,
                   b: int, hk: int, d: int) -> tuple[float, float, list[float]]:
    if not entries:
        return -math.inf, 0.0, [0.0] * d
    m = -math.inf
    l = 0.0
    u = [0.0] * d
    for ki, score in entries:
        new_m = max(m, score)
        old_scale = 0.0 if l == 0.0 else math.exp(m - new_m)
        weight = math.exp(score - new_m)
        l = old_scale * l + weight
        u = [old_scale * u[x] + weight * v.at(b, hk, ki, x) for x in range(d)]
        m = new_m
    return m, l, u


def _merge_states(states: Sequence[tuple[float, float, list[float]]], d: int
                  ) -> tuple[list[float], float, float, float]:
    nonempty = [(m, l, u) for m, l, u in states if l > 0.0]
    if not nonempty:
        return [0.0] * d, -math.inf, 0.0, -math.inf
    m = max(state[0] for state in nonempty)
    l = math.fsum(math.exp(mi - m) * li for mi, li, _ in nonempty)
    u = [math.fsum(math.exp(mi - m) * ui[x] for mi, _, ui in nonempty)
         for x in range(d)]
    return [x / l for x in u], m, l, m + math.log(l)


def attention_online(q: Tensor4, k: Tensor4, v: Tensor4, *, key_ranges: Sequence[tuple[int, int]],
                     causal: bool = False,
                     masked: Sequence[Sequence[Sequence[Sequence[bool]]]] | None = None,
                     selected_blocks: Sequence[Sequence[Sequence[Sequence[int]]]] | None = None,
                     block_size: int = 1) -> dict[str, object]:
    """FP64 implementation of independent block partials and stable merging."""
    bsz, hq, hkv, sq, sk, d = _validate(q, k, v, causal)
    _validate_row_mask(masked, bsz, hq, sq, sk)
    _validate_selection_shape(selected_blocks, bsz, hq, sq)
    if not key_ranges:
        raise ValueError("at least one key range is required")
    cursor = 0
    for lo, hi in key_ranges:
        if type(lo) is not int or type(hi) is not int or lo != cursor or hi < lo or hi > sk:
            raise ValueError("key ranges must be contiguous, ordered, and cover from zero")
        cursor = hi
    if cursor != sk:
        raise ValueError("key ranges must cover the full key sequence")
    if selected_blocks is not None and (type(block_size) is not int or block_size <= 0):
        raise ValueError("selected-block attention requires a positive integer block size")
    nblocks = math.ceil(sk / block_size) if selected_blocks is not None else 0
    output: list[float] = []
    row_stats: list[dict[str, float | int]] = []
    for b in range(bsz):
        for h in range(hq):
            hk = h // (hq // hkv)
            for qi in range(sq):
                selected = _selected_for_row(selected_blocks, b, h, qi, nblocks)
                entries = _row_keys(q, k, b, h, qi, causal=causal, mask=masked,
                                    selected=selected, block_size=block_size)
                states = []
                for lo, hi in key_ranges:
                    states.append(_partial_state([(ki, score) for ki, score in entries
                                                  if lo <= ki < hi], v, b, hk, d))
                row_output, m, l, lse = _merge_states(states, d)
                output.extend(row_output)
                row_stats.append({"batch": b, "query_head": h, "query_row": qi,
                                  "max_score": m, "normalizer": l, "logsumexp": lse,
                                  "valid_keys": len(entries)})
    return {"shape": (bsz, hq, sq, d), "output": tuple(output), "rows": row_stats}


def write_f16le_bundle(path: Path, q: Tensor4, k: Tensor4, v: Tensor4,
                       metadata_path: Path) -> None:
    if any(len(t.values) == 0 for t in (q, k, v)):
        raise ValueError("empty tensors are not supported")
    cursor = 0
    records = {}
    with path.open("wb") as out:
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            start = cursor
            for value in tensor.values:
                raw = struct.pack("<e", value)
                out.write(raw)
                cursor += len(raw)
            records[name] = {"shape": tensor.shape, "dtype": "float16_le",
                             "byte_offset": start, "byte_length": cursor - start}
    metadata_path.write_text(json.dumps({"format": "KERNELWIKI-ATTENTION-INPUTS/1",
                                         "arrays": records,
                                         "total_bytes": cursor}, indent=2) + "\n",
                             encoding="utf-8")
