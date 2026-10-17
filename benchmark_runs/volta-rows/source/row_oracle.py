#!/usr/bin/env python3
"""CPU-only represented-input and independent FP64 oracle for KWV-05 rows.

The files deliberately contain the exact FP32 bit patterns passed to CUDA.
The reference is calculated from those decoded values, never from a seed.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any

INPUT_MAGIC = b"KWVROWS1"
REFERENCE_MAGIC = b"KWVRREF1"
INPUT_FORMAT = "KERNELWIKI-ROW-INPUT/1"
REFERENCE_FORMAT = "KERNELWIKI-ROW-REFERENCE/1"
ORACLE_FORMAT = "KERNELWIKI-CPU-ORACLE/1"
VERSION = 1
ID_BYTES = 32
MAX_COLS = 8193
HEADER = struct.Struct("<I32s14I3f")
# Exactly representable binary32 thresholds avoid a discrepancy between the
# JSON contract and the fixed-width input header consumed by the native runner.
ATOL = 2.0 ** -30
RTOL = 2.0 ** -14


class ContractError(ValueError):
    pass


@dataclass
class Case:
    case_id: str
    operation: str
    rows: int
    cols: int
    x_row_stride: int
    y_row_stride: int
    mask_row_stride: int
    x_offset: int
    y_offset: int
    mask_offset: int
    gamma_offset: int
    epsilon: float
    x: list[float]
    mask: list[int]
    gamma: list[float]
    mask_enabled: bool = False

    @property
    def logical_count(self) -> int:
        return self.rows * self.cols


def _f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise ContractError(f"value cannot be represented as FP32: {value}") from exc


def _case_id_bytes(case_id: str) -> bytes:
    try:
        raw = case_id.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ContractError("case_id must contain ASCII characters only") from exc
    if not raw or len(raw) >= ID_BYTES or b"\0" in raw:
        raise ContractError(f"case_id must be 1..{ID_BYTES - 1} ASCII bytes without NUL")
    return raw + bytes(ID_BYTES - len(raw))


def _decode_id(raw: bytes) -> str:
    try:
        return raw.split(b"\0", 1)[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ContractError("case_id is not ASCII") from exc


def validate_case(case: Case) -> None:
    if case.operation not in {"softmax", "rmsnorm"}:
        raise ContractError("operation must be softmax or rmsnorm")
    _case_id_bytes(case.case_id)
    values = (case.rows, case.cols, case.x_row_stride, case.y_row_stride,
              case.mask_row_stride, case.x_offset, case.y_offset,
              case.mask_offset, case.gamma_offset)
    if any(type(value) is not int or value < 0 or value > 0x7fffffff for value in values):
        raise ContractError("dimensions, strides, and offsets must be nonnegative 31-bit integers")
    if not 1 <= case.cols <= MAX_COLS:
        raise ContractError(f"N must be in [1, {MAX_COLS}]; zero-row-count is the empty case")
    if case.x_row_stride < case.cols or case.y_row_stride < case.cols:
        raise ContractError("x/y row strides must be at least N")
    if case.mask_enabled and case.mask_row_stride < case.cols:
        raise ContractError("mask row stride must be at least N")
    if case.rows == 0 and (case.x_offset or case.y_offset or case.mask_offset):
        raise ContractError("zero-row cases require zero base offsets")
    if case.rows > 0:
        x_required = case.x_offset + (case.rows - 1) * case.x_row_stride + case.cols
        y_required = case.y_offset + (case.rows - 1) * case.y_row_stride + case.cols
        if len(case.x) != x_required:
            raise ContractError(f"x extent must be exactly {x_required} elements")
        if case.mask_enabled:
            mask_required = case.mask_offset + (case.rows - 1) * case.mask_row_stride + case.cols
            if case.mask_row_stride < case.cols or len(case.mask) != mask_required:
                raise ContractError(f"mask extent/stride must cover exactly {mask_required} elements")
        elif case.mask or case.mask_offset:
            raise ContractError("unmasked case must not carry a mask or mask offset")
    elif case.x or case.mask:
        raise ContractError("zero-row case must have empty x/mask arrays")
    if case.operation == "rmsnorm":
        if case.mask_enabled or case.mask:
            raise ContractError("RMSNorm contract does not include a mask")
        if len(case.gamma) != case.gamma_offset + case.cols:
            raise ContractError("gamma extent must be exactly gamma_offset + N")
        epsilon = _f32(case.epsilon)
        if not math.isfinite(epsilon) or not (1.0e-5 <= epsilon <= 1.0e-2):
            raise ContractError("RMSNorm epsilon must be finite FP32 in [1e-5, 1e-2]")
        if any(not math.isfinite(_f32(value)) or abs(_f32(value)) > 4.0 for value in case.gamma):
            raise ContractError("gamma values must be finite FP32 with absolute value <= 4")
    elif case.gamma or case.gamma_offset or case.epsilon != 0:
        raise ContractError("softmax case must not carry gamma or epsilon")
    for row in range(case.rows):
        for col in range(case.cols):
            value = _f32(case.x[case.x_offset + row * case.x_row_stride + col])
            if not math.isfinite(value):
                raise ContractError("logical input values must be finite FP32")
            bound = 16.0 if case.operation == "softmax" else 64.0
            if abs(value) > bound:
                raise ContractError(f"logical input absolute value exceeds declared {bound:g} bound")
    if case.operation == "softmax" and case.mask_enabled:
        for row in range(case.rows):
            for col in range(case.cols):
                value = case.mask[case.mask_offset + row * case.mask_row_stride + col]
                if type(value) is not int or value not in (0, 1):
                    raise ContractError("mask bytes must be 0 or 1")


def _pack_array_f32(values: list[float]) -> bytes:
    try:
        return struct.pack("<" + "f" * len(values), *values) if values else b""
    except (OverflowError, struct.error) as exc:
        raise ContractError(f"FP32 array contains unrepresentable value: {exc}") from exc


def encode_case(case: Case) -> bytes:
    validate_case(case)
    flags = 1 if case.mask_enabled else 0
    values = [VERSION, _case_id_bytes(case.case_id),
              1 if case.operation == "softmax" else 2,
              case.rows, case.cols, case.x_row_stride, case.y_row_stride,
              case.mask_row_stride, case.x_offset, case.y_offset,
              case.mask_offset, case.gamma_offset, len(case.x), len(case.mask),
              len(case.gamma), flags, _f32(case.epsilon), ATOL, RTOL]
    # The struct has one version integer, fixed id, fourteen integers, and
    # three FP32 values. Keep field order mirrored by the C++ runner.
    header = HEADER.pack(*values)
    return (header + _pack_array_f32(case.x) + bytes(case.mask) +
            _pack_array_f32(case.gamma))


def decode_case(raw: bytes, offset: int = 0) -> tuple[Case, int, float, float]:
    if len(raw) - offset < HEADER.size:
        raise ContractError("truncated row case header")
    unpacked = HEADER.unpack_from(raw, offset)
    offset += HEADER.size
    (version, raw_id, op, rows, cols, x_stride, y_stride, mask_stride,
     x_offset, y_offset, mask_offset, gamma_offset, x_count, mask_count,
     gamma_count, flags, epsilon, atol, rtol) = unpacked
    if version != VERSION or op not in (1, 2) or flags & ~1:
        raise ContractError("unsupported row case version, operation, or flags")
    byte_count = x_count * 4 + mask_count + gamma_count * 4
    if byte_count > len(raw) - offset:
        raise ContractError("truncated row case payload")
    x_end = offset + x_count * 4
    x = list(struct.unpack_from("<" + "f" * x_count, raw, offset)) if x_count else []
    mask_end = x_end + mask_count
    mask = list(raw[x_end:mask_end])
    gamma_end = mask_end + gamma_count * 4
    gamma = (list(struct.unpack_from("<" + "f" * gamma_count, raw, mask_end))
             if gamma_count else [])
    case = Case(_decode_id(raw_id), "softmax" if op == 1 else "rmsnorm",
                rows, cols, x_stride, y_stride, mask_stride, x_offset,
                y_offset, mask_offset, gamma_offset, epsilon, x, mask, gamma,
                bool(flags & 1))
    validate_case(case)
    return case, gamma_end, atol, rtol


def encode_bundle(cases: list[Case]) -> bytes:
    if not cases:
        raise ContractError("input bundle must contain at least one case")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ContractError("case IDs must be unique within an input bundle")
    return INPUT_MAGIC + struct.pack("<I", len(cases)) + b"".join(encode_case(c) for c in cases)


def decode_bundle(raw: bytes) -> list[tuple[Case, float, float]]:
    if len(raw) < 12 or raw[:8] != INPUT_MAGIC:
        raise ContractError("row input magic is invalid")
    count = struct.unpack_from("<I", raw, 8)[0]
    if count == 0 or count > 10000:
        raise ContractError("row input case count is invalid")
    offset = 12
    result = []
    for _ in range(count):
        case, offset, atol, rtol = decode_case(raw, offset)
        if not (math.isfinite(atol) and math.isfinite(rtol) and atol >= 0 and rtol >= 0):
            raise ContractError("numerical tolerance must be finite and nonnegative")
        if atol != ATOL or rtol != RTOL:
            raise ContractError("input bundle tolerances differ from the fixed FP32 comparison contract")
        result.append((case, atol, rtol))
    if offset != len(raw):
        raise ContractError("row input bundle has trailing bytes")
    ids = [case.case_id for case, _, _ in result]
    if len(ids) != len(set(ids)):
        raise ContractError("case IDs must be unique within an input bundle")
    return result


def compute_reference(case: Case) -> list[float]:
    validate_case(case)
    out: list[float] = []
    if case.operation == "softmax":
        for row in range(case.rows):
            values = [float(case.x[case.x_offset + row * case.x_row_stride + col])
                      for col in range(case.cols)]
            active = [col for col in range(case.cols)
                      if not case.mask_enabled or
                      case.mask[case.mask_offset + row * case.mask_row_stride + col] != 0]
            if not active:
                out.extend([0.0] * case.cols)
                continue
            maximum = max(values[col] for col in active)
            exps = [math.exp(values[col] - maximum) for col in active]
            denominator = math.fsum(exps)
            row_out = [0.0] * case.cols
            for col, value in zip(active, exps):
                row_out[col] = value / denominator
            out.extend(row_out)
        return out
    gamma = [float(case.gamma[case.gamma_offset + col]) for col in range(case.cols)]
    epsilon = float(_f32(case.epsilon))
    for row in range(case.rows):
        values = [float(case.x[case.x_offset + row * case.x_row_stride + col])
                  for col in range(case.cols)]
        mean_square = math.fsum(value * value for value in values) / case.cols
        inv = 1.0 / math.sqrt(epsilon + mean_square)
        out.extend(gamma[col] * values[col] * inv for col in range(case.cols))
    return out


def encode_reference(entries: list[tuple[str, list[float]]]) -> bytes:
    if not entries:
        raise ContractError("reference bundle must contain at least one case")
    if len({case_id for case_id, _ in entries}) != len(entries):
        raise ContractError("reference case IDs must be unique")
    parts = [REFERENCE_MAGIC, struct.pack("<I", len(entries))]
    for case_id, values in entries:
        _case_id_bytes(case_id)
        parts.extend((_case_id_bytes(case_id), struct.pack("<Q", len(values)),
                      struct.pack("<" + "d" * len(values), *values) if values else b""))
    return b"".join(parts)


def decode_reference(raw: bytes) -> dict[str, list[float]]:
    if len(raw) < 12 or raw[:8] != REFERENCE_MAGIC:
        raise ContractError("row reference magic is invalid")
    count = struct.unpack_from("<I", raw, 8)[0]
    offset = 12
    result: dict[str, list[float]] = {}
    for _ in range(count):
        if len(raw) - offset < ID_BYTES + 8:
            raise ContractError("truncated reference case header")
        case_id = _decode_id(raw[offset:offset + ID_BYTES]); offset += ID_BYTES
        value_count = struct.unpack_from("<Q", raw, offset)[0]; offset += 8
        size = value_count * 8
        if size > len(raw) - offset:
            raise ContractError("truncated reference values")
        if case_id in result:
            raise ContractError("duplicate reference case ID")
        result[case_id] = (list(struct.unpack_from("<" + "d" * value_count, raw, offset))
                           if value_count else [])
        offset += size
    if count == 0 or offset != len(raw):
        raise ContractError("empty reference bundle or trailing bytes")
    return result


def check_output(case: Case, expected: list[float], candidate: list[float],
                 atol: float = ATOL, rtol: float = RTOL) -> dict[str, Any]:
    if len(expected) != case.logical_count or len(candidate) != case.logical_count:
        raise ContractError("logical candidate/reference extent does not match rows*N")
    checked = failed = nonfinite = 0
    max_error = max_tolerance = max_ratio = 0.0
    for actual, reference in zip(candidate, expected):
        checked += 1
        if not math.isfinite(actual):
            nonfinite += 1
            failed += 1
            continue
        error = abs(actual - reference)
        tolerance = atol + rtol * abs(reference)
        max_error = max(max_error, error)
        max_tolerance = max(max_tolerance, tolerance)
        ratio = error / tolerance if tolerance else (0.0 if error == 0.0 else math.inf)
        max_ratio = max(max_ratio, ratio)
        if error > tolerance:
            failed += 1
    return {"case_id": case.case_id, "checked_outputs": checked,
            "failed_outputs": failed, "nonfinite_outputs": nonfinite,
            "max_abs_error": max_error,
            "max_admission_tolerance": max_tolerance,
            "max_tolerance_ratio": max_ratio,
            "atol": atol, "rtol": rtol, "passed": failed == 0}


def oracle_record(input_bytes: bytes, reference_bytes: bytes) -> dict[str, Any]:
    cases = decode_bundle(input_bytes)
    references = decode_reference(reference_bytes)
    checks = []
    for case, atol, rtol in cases:
        expected = compute_reference(case)
        stored = references.get(case.case_id)
        if stored is None or len(stored) != len(expected):
            raise ContractError(f"reference missing/mismatched for {case.case_id}")
        # Generation self-check proves the serialized FP64 file contains the
        # independently computed oracle values in the exact decoded input domain.
        if any(not math.isfinite(x) or abs(x - y) > 1e-14 * max(1.0, abs(y))
               for x, y in zip(stored, expected)):
            raise ContractError(f"serialized reference differs from FP64 oracle for {case.case_id}")
        checks.append({"case_id": case.case_id, "operation": case.operation,
                       "rows": case.rows, "cols": case.cols,
                       "reference_values": len(expected), "passed": True,
                       "max_abs_error": 0.0, "reference_max_abs": max(map(abs, expected), default=0.0),
                       "atol": atol, "rtol": rtol})
    if set(references) != {case.case_id for case, _, _ in cases}:
        raise ContractError("reference bundle case IDs differ from input bundle")
    return {"format": ORACLE_FORMAT, "passed": True, "oracle": "Python binary64 over exact decoded FP32 values",
            "checks": checks,
            "represented_input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_bytes).hexdigest()}
