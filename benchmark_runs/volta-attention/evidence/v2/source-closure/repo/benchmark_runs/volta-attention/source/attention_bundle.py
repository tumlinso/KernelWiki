#!/usr/bin/env python3
"""Portable little-endian represented-input and FP64-reference bundle I/O."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Sequence

from attention_oracle import Tensor4

INPUT_MAGIC = b"KWVATTN1"
REFERENCE_MAGIC = b"KWVREF01"
VERSION = 1
INPUT_HEADER = struct.Struct("<8sII")
INPUT_RECORD = struct.Struct("<64s10I12Q")
REFERENCE_HEADER = struct.Struct("<8sII")
REFERENCE_RECORD = struct.Struct("<64s8I4Q")
REFERENCE_ROW = struct.Struct("<dddQ")


@dataclass(frozen=True)
class InputCase:
    name: str
    q: Tensor4
    k: Tensor4
    v: Tensor4
    causal: bool
    block_size: int
    selected_blocks: Sequence[Sequence[Sequence[Sequence[int]]]] | None
    masked: Sequence[Sequence[Sequence[Sequence[bool]]]] | None


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    shape: tuple[int, int, int, int]
    kv_shape: tuple[int, int, int, int]
    output: tuple[float, ...]
    rows: tuple[dict[str, Any], ...]


def _pad_name(name: str) -> bytes:
    encoded = name.encode("ascii")
    if not encoded or len(encoded) >= 64 or b"\0" in encoded:
        raise ValueError("case name must be 1..63 ASCII bytes without NUL")
    return encoded + bytes(64 - len(encoded))


def _pack_values(fmt: str, values) -> bytes:
    item = struct.Struct("<" + fmt)
    result = bytearray()
    for value in values:
        result.extend(item.pack(value))
    return bytes(result)


def _flatten_mask(mask, b: int, h: int, s: int, k: int) -> list[int] | None:
    if mask is None:
        return None
    if len(mask) != b:
        raise ValueError("mask batch shape differs from Q")
    values = []
    for batch in mask:
        if len(batch) != h:
            raise ValueError("mask head shape differs from Q")
        for head in batch:
            if len(head) != s:
                raise ValueError("mask query shape differs from Q")
            for row in head:
                if len(row) != k or any(type(x) is not bool for x in row):
                    raise ValueError("mask key shape/type differs from Q/K")
                values.extend(int(x) for x in row)
    return values


def _selection_arrays(selected, b: int, h: int, s: int, sk: int, block: int):
    if selected is None:
        return None
    if block <= 0 or len(selected) != b or any(len(batch) != h for batch in selected):
        raise ValueError("selected-block metadata shape or block size is invalid")
    if any(len(head) != s for batch in selected for head in batch):
        raise ValueError("selected-block query shape differs from Q")
    max_count = max(1, max((len(ids) for batch in selected for head in batch for ids in head), default=0))
    counts, ids_flat = [], []
    block_count = (sk + block - 1) // block
    for batch in selected:
        for head in batch:
            for row in head:
                if any(type(i) is not int for i in row) or len(set(row)) != len(row):
                    raise ValueError("selected block IDs must be unique integers")
                if any(i < 0 or i >= block_count for i in row):
                    raise ValueError("selected block ID outside sequence")
                counts.append(len(row))
                ids_flat.extend(row)
                ids_flat.extend([-1] * (max_count - len(row)))
    return max_count, counts, ids_flat


def write_input_bundle(path: Path, cases: Sequence[InputCase]) -> list[dict[str, Any]]:
    if not cases:
        raise ValueError("input bundle requires at least one case")
    table_size = INPUT_HEADER.size + len(cases) * INPUT_RECORD.size
    data = bytearray(table_size)
    records = []
    for case in cases:
        b, hq, sq, d = case.q.shape
        bk, hkv, sk, dk = case.k.shape
        if (bk != b or case.v.shape != (b, hkv, sk, d) or dk != d or
                hq % hkv or case.causal and sq > sk):
            raise ValueError(f"{case.name}: incompatible tensor dimensions")
        row_count = b * hq * sq
        flags = int(case.causal)
        mask = _flatten_mask(case.masked, b, hq, sq, sk)
        if mask is not None:
            flags |= 4
        selection = _selection_arrays(case.selected_blocks, b, hq, sq, sk,
                                      case.block_size)
        if selection is None:
            max_selected, counts, selected_ids = 0, [], []
        else:
            flags |= 2
            max_selected, counts, selected_ids = selection
        if case.selected_blocks is not None:
            flags |= 2
        if case.causal:
            flags |= 1

        def append(payload: bytes) -> tuple[int, int]:
            offset = len(data)
            data.extend(payload)
            return offset, len(payload)

        qoff, qlen = append(_pack_values("e", case.q.values))
        koff, klen = append(_pack_values("e", case.k.values))
        voff, vlen = append(_pack_values("e", case.v.values))
        if mask is None:
            moff, mlen = 0, 0
        else:
            moff, mlen = append(bytes(mask))
        if selected_ids:
            idoff, idlen = append(_pack_values("i", selected_ids))
            countoff, countlen = append(_pack_values("i", counts))
        elif case.selected_blocks is not None:
            # Preserve an explicit all-empty selection with a positive slot
            # capacity and a zero count for every row.
            idoff, idlen = append(_pack_values("i", [-1] * (row_count * max_selected)))
            countoff, countlen = append(_pack_values("i", [0] * row_count))
        else:
            idoff, idlen, countoff, countlen = 0, 0, 0, 0
        ints = (b, hq, hkv, sq, sk, d, flags, case.block_size,
                max_selected, 0)
        quads = (qoff, qlen // 2, koff, klen // 2, voff, vlen // 2,
                 moff, mlen, idoff, idlen // 4, countoff, countlen // 4)
        records.append(INPUT_RECORD.pack(_pad_name(case.name), *ints, *quads))
        records[-1] = {"_packed": records[-1], "id": case.name,
                       "shape": {"q": case.q.shape, "k": case.k.shape,
                                 "v": case.v.shape},
                       "flags": flags, "block_size": case.block_size,
                       "max_selected_blocks": max_selected,
                       "offsets": {"q": qoff, "k": koff, "v": voff,
                                   "mask": moff, "selected_ids": idoff,
                                   "selected_counts": countoff}}
    data[:INPUT_HEADER.size] = INPUT_HEADER.pack(INPUT_MAGIC, VERSION, len(cases))
    for i, record in enumerate(records):
        start = INPUT_HEADER.size + i * INPUT_RECORD.size
        data[start:start + INPUT_RECORD.size] = record["_packed"]
        del record["_packed"]
    path.write_bytes(data)
    return records


def write_reference_bundle(path: Path, cases: Sequence[ReferenceCase]) -> list[dict[str, Any]]:
    if not cases:
        raise ValueError("reference bundle requires at least one case")
    table_size = REFERENCE_HEADER.size + len(cases) * REFERENCE_RECORD.size
    data = bytearray(table_size)
    records = []
    for case in cases:
        b, h, sq, d = case.shape
        bk, hkv, sk, dk = case.kv_shape
        if (bk != b or dk != d or len(case.output) != b * h * sq * d or
                len(case.rows) != b * h * sq):
            raise ValueError(f"{case.name}: reference output/row count differs from shape")
        outoff = len(data)
        data.extend(_pack_values("d", case.output))
        rowoff = len(data)
        for row in case.rows:
            data.extend(REFERENCE_ROW.pack(float(row["max_score"]),
                                           float(row["normalizer"]),
                                           float(row["logsumexp"]),
                                           int(row["valid_keys"])))
        ints = (b, h, hkv, sq, sk, d, 0, 0)
        quads = (outoff, len(case.output), rowoff, len(case.rows))
        records.append(REFERENCE_RECORD.pack(_pad_name(case.name), *ints, *quads))
    data[:REFERENCE_HEADER.size] = REFERENCE_HEADER.pack(REFERENCE_MAGIC, VERSION,
                                                         len(cases))
    for i, record in enumerate(records):
        start = REFERENCE_HEADER.size + i * REFERENCE_RECORD.size
        data[start:start + REFERENCE_RECORD.size] = record
    path.write_bytes(data)
    return [{"id": case.name, "shape": case.shape,
             "output_count": len(case.output), "row_count": len(case.rows)}
            for case in cases]


def _name(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii")


def read_input_bundle(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if len(raw) < INPUT_HEADER.size:
        raise ValueError("input bundle header is truncated")
    magic, version, ncases = INPUT_HEADER.unpack_from(raw)
    if magic != INPUT_MAGIC or version != VERSION or ncases <= 0:
        raise ValueError("unsupported input bundle header")
    table_end = INPUT_HEADER.size + ncases * INPUT_RECORD.size
    if table_end > len(raw):
        raise ValueError("input case table is truncated")
    result = []
    for index in range(ncases):
        rec = INPUT_RECORD.unpack_from(raw, INPUT_HEADER.size + index * INPUT_RECORD.size)
        name = _name(rec[0])
        b, hq, hkv, sq, sk, d, flags, block_size, maxsel, reserved = rec[1:11]
        qoff, qcount, koff, kcount, voff, vcount, moff, mcount, idoff, idcount, coff, ccount = rec[11:23]
        if (reserved != 0 or not name or min(b, hq, hkv, sq, sk, d) <= 0 or
                hq % hkv or flags & ~7 or flags & 1 and sq > sk):
            raise ValueError(f"{name}: invalid input case metadata")
        rows = b * hq * sq
        if qcount != rows * d or kcount != b * hkv * sk * d or vcount != kcount:
            raise ValueError(f"{name}: tensor element count disagrees with shape")

        def section(offset: int, count: int, width: int, label: str) -> memoryview:
            if count == 0:
                if offset != 0:
                    raise ValueError(f"{name}: empty {label} has a nonzero offset")
                return memoryview(b"")
            length = count * width
            if offset < table_end or offset + length > len(raw):
                raise ValueError(f"{name}: {label} exceeds input bundle bounds")
            return memoryview(raw)[offset:offset + length]

        def halfs(offset: int, count: int, label: str) -> tuple[float, ...]:
            blob = section(offset, count, 2, label)
            return tuple(item[0] for item in struct.iter_unpack("<e", blob))

        qdata = halfs(qoff, qcount, "Q")
        kdata = halfs(koff, kcount, "K")
        vdata = halfs(voff, vcount, "V")
        mask_values = list(section(moff, mcount, 1, "mask"))
        if bool(flags & 4) != bool(mcount) or mcount not in (0, rows * sk):
            raise ValueError(f"{name}: mask metadata is inconsistent")
        masked = None
        if mask_values:
            if any(x not in (0, 1) for x in mask_values):
                raise ValueError(f"{name}: mask bytes must be 0 or 1")
            masked = []
            cursor = 0
            for _b in range(b):
                batch = []
                for _h in range(hq):
                    head = []
                    for _q in range(sq):
                        head.append([bool(x) for x in mask_values[cursor:cursor + sk]])
                        cursor += sk
                    batch.append(head)
                masked.append(batch)
        selected = None
        if flags & 2:
            if block_size <= 0 or maxsel <= 0 or idcount != rows * maxsel or ccount != rows:
                raise ValueError(f"{name}: selected-block metadata is inconsistent")
            id_values = [x[0] for x in struct.iter_unpack("<i", section(idoff, idcount, 4, "selected IDs"))]
            counts = [x[0] for x in struct.iter_unpack("<i", section(coff, ccount, 4, "selected counts"))]
            nblocks = (sk + block_size - 1) // block_size
            selected = []
            for _b in range(b):
                batch = []
                for _h in range(hq):
                    head = []
                    for _q in range(sq):
                        slot_row = (len(selected) * hq * sq + len(batch) * sq + len(head))
                        count = counts[slot_row]
                        base = slot_row * maxsel
                        ids = id_values[base:base + count]
                        if (count < 0 or count > maxsel or len(set(ids)) != len(ids) or
                                any(i < 0 or i >= nblocks for i in ids)):
                            raise ValueError(f"{name}: invalid selected block row")
                        head.append(ids)
                    batch.append(head)
                selected.append(batch)
        elif any((idoff, idcount, coff, ccount, maxsel)):
            raise ValueError(f"{name}: unselected case contains selected-block data")
        result.append({"id": name, "shape": (b, hq, sq, d),
                       "kv_shape": (b, hkv, sk, d), "causal": bool(flags & 1),
                       "block_size": block_size, "selected_blocks": selected,
                       "masked": masked,
                       "q": Tensor4((b, hq, sq, d), qdata),
                       "k": Tensor4((b, hkv, sk, d), kdata),
                       "v": Tensor4((b, hkv, sk, d), vdata)})
    return result


def read_reference_bundle(path: Path, inputs: Sequence[dict[str, Any]]) -> dict[str, ReferenceCase]:
    raw = path.read_bytes()
    if len(raw) < REFERENCE_HEADER.size:
        raise ValueError("reference bundle header is truncated")
    magic, version, ncases = REFERENCE_HEADER.unpack_from(raw)
    if magic != REFERENCE_MAGIC or version != VERSION or ncases != len(inputs):
        raise ValueError("reference bundle does not match input suite")
    table_end = REFERENCE_HEADER.size + ncases * REFERENCE_RECORD.size
    if table_end > len(raw):
        raise ValueError("reference case table is truncated")
    results = {}
    for index, input_case in enumerate(inputs):
        rec = REFERENCE_RECORD.unpack_from(raw, REFERENCE_HEADER.size + index * REFERENCE_RECORD.size)
        name = _name(rec[0])
        b, h, hkv, sq, sk, d, reserved1, reserved2 = rec[1:9]
        outoff, outcount, rowoff, rowcount = rec[9:13]
        if (name != input_case["id"] or (b, h, sq, d) != input_case["shape"] or
                (hkv, sk) != (input_case["kv_shape"][1], input_case["kv_shape"][2]) or
                reserved1 != 0 or reserved2 != 0 or
                outcount != b * h * sq * d or rowcount != b * h * sq):
            raise ValueError(f"{name}: reference metadata differs from input suite")
        if (outoff < table_end or outoff + outcount * 8 > len(raw) or
                rowoff < table_end or rowoff + rowcount * REFERENCE_ROW.size > len(raw)):
            raise ValueError(f"{name}: reference arrays exceed bundle bounds")
        output = tuple(x[0] for x in struct.iter_unpack("<d", raw[outoff:outoff + outcount * 8]))
        rows = []
        for m, normalizer, lse, valid in struct.iter_unpack(
                "<dddQ", raw[rowoff:rowoff + rowcount * REFERENCE_ROW.size]):
            rows.append({"max_score": m, "normalizer": normalizer,
                         "logsumexp": lse, "valid_keys": valid})
        results[name] = ReferenceCase(name, (b, h, sq, d), (b, hkv, sk, d),
                                      output, tuple(rows))
    return results
