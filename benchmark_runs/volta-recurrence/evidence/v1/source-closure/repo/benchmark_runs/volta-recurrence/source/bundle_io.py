"""Portable represented-input/reference bundle for the KWV-08 native runner."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"KWV08BND"
VERSION = 1
HEADER = struct.Struct("<8sII64s")
ATOL = 2.0e-5
RTOL = 2.0e-4
SENTINEL = np.float32(12345.25)


@dataclass
class BundleCase:
    case_id: str
    keys: np.ndarray
    values: np.ndarray
    queries: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    lengths: np.ndarray
    initial_states: np.ndarray
    expected_outputs: np.ndarray
    expected_final_states: np.ndarray

    @property
    def shape(self):
        t, batch, dk = self.keys.shape
        dv = self.values.shape[2]
        return t, batch, dv, dk


def _append_array(chunks: list[bytes], array: np.ndarray, dtype: str) -> None:
    canonical = np.asarray(array, dtype=np.dtype(dtype), order="C")
    chunks.append(canonical.tobytes(order="C"))


def encode_bundle(cases: list[BundleCase]) -> tuple[bytes, dict]:
    if not cases:
        raise ValueError("bundle must contain at least one case")
    payload: list[bytes] = []
    metadata = []
    seen = set()
    for case in cases:
        case_id = case.case_id.encode("utf-8")
        if not case_id or len(case_id) > 255 or case.case_id in seen:
            raise ValueError("case IDs must be unique UTF-8 strings of 1..255 bytes")
        seen.add(case.case_id)
        t, batch, dv, dk = case.shape
        if min(t, batch, dv, dk) <= 0:
            raise ValueError("case dimensions must be positive")
        payload.append(struct.pack("<I", len(case_id)))
        payload.append(case_id)
        payload.append(struct.pack("<4I", t, batch, dv, dk))
        _append_array(payload, case.keys, "<f4")
        _append_array(payload, case.values, "<f4")
        _append_array(payload, case.queries, "<f4")
        _append_array(payload, case.alpha, "<f4")
        _append_array(payload, case.beta, "<f4")
        _append_array(payload, case.lengths, "<i4")
        _append_array(payload, case.initial_states, "<f4")
        _append_array(payload, case.expected_outputs, "<f8")
        _append_array(payload, case.expected_final_states, "<f8")
        metadata.append({
            "case_id": case.case_id,
            "T": t,
            "B": batch,
            "d_v": dv,
            "d_k": dk,
            "live_steps": int(np.sum(case.lengths)),
            "length_min": int(np.min(case.lengths)),
            "length_max": int(np.max(case.lengths)),
            "padded_input_nan_count": int(
                np.count_nonzero(~np.isfinite(case.keys))
                + np.count_nonzero(~np.isfinite(case.values))
                + np.count_nonzero(~np.isfinite(case.queries))
                + np.count_nonzero(~np.isfinite(case.alpha))
                + np.count_nonzero(~np.isfinite(case.beta))
            ),
        })
    payload_bytes = b"".join(payload)
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    binary = HEADER.pack(MAGIC, VERSION, len(cases), payload_sha.encode("ascii")) + payload_bytes
    manifest = {
        "format": "KERNELWIKI-VOLTA-RECURRENCE-BUNDLE/1",
        "bundle_payload_sha256": payload_sha,
        "bundle_file_sha256": hashlib.sha256(binary).hexdigest(),
        "bundle_bytes": len(binary),
        "byte_order": "little-endian IEEE754 binary32/binary64 and signed int32",
        "case_count": len(cases),
        "cases": metadata,
        "numerical_contract": {
            "operator": "D=alpha*S_previous; e=v-D@k; S=D+beta*outer(e,k); o=S@q",
            "state_orientation": "[d_v,d_k]",
            "output_state": "post-update",
            "inputs_and_state": "FP32 represented values; independent oracle uses FP64",
            "key_policy": "active test keys are generated unit length; runner never normalizes them",
            "gates": "active alpha,beta in [0,1], including exact 0 and near-one values",
            "active_input_policy": "finite and bounded; NaN values intentionally mark only padded steps",
            "atol": ATOL,
            "rtol": RTOL,
            "comparison": "abs(actual-reference) <= atol + rtol*abs(reference), on every live output and final-state element",
            "output_padding": "runner initializes outputs to SENTINEL and requires padded locations to remain unchanged",
        },
    }
    return binary, manifest


def _take(payload: memoryview, offset: int, count: int, dtype: str):
    dt = np.dtype(dtype)
    nbytes = count * dt.itemsize
    end = offset + nbytes
    if count < 0 or end > len(payload):
        raise ValueError("truncated recurrence bundle payload")
    return np.frombuffer(payload[offset:end], dtype=dt).copy(), end


def decode_bundle(binary: bytes) -> tuple[list[BundleCase], dict]:
    if len(binary) < HEADER.size:
        raise ValueError("truncated recurrence bundle header")
    magic, version, case_count, payload_sha_bytes = HEADER.unpack_from(binary)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported recurrence bundle magic/version")
    payload = memoryview(binary)[HEADER.size:]
    payload_sha = payload_sha_bytes.decode("ascii")
    if hashlib.sha256(payload).hexdigest() != payload_sha:
        raise ValueError("recurrence bundle payload SHA-256 mismatch")
    offset = 0
    cases = []
    for _ in range(case_count):
        name_len_data, offset = _take(payload, offset, 1, "<u4")
        name_len = int(name_len_data[0])
        if name_len < 1 or offset + name_len > len(payload):
            raise ValueError("invalid recurrence case ID length")
        case_id = bytes(payload[offset:offset + name_len]).decode("utf-8")
        offset += name_len
        dims, offset = _take(payload, offset, 4, "<u4")
        t, batch, dv, dk = map(int, dims)
        if min(t, batch, dv, dk) <= 0:
            raise ValueError("invalid recurrence case dimensions")
        keys, offset = _take(payload, offset, t * batch * dk, "<f4")
        values, offset = _take(payload, offset, t * batch * dv, "<f4")
        queries, offset = _take(payload, offset, t * batch * dk, "<f4")
        alpha, offset = _take(payload, offset, t * batch, "<f4")
        beta, offset = _take(payload, offset, t * batch, "<f4")
        lengths, offset = _take(payload, offset, batch, "<i4")
        initial, offset = _take(payload, offset, batch * dv * dk, "<f4")
        expected_outputs, offset = _take(payload, offset, t * batch * dv, "<f8")
        expected_final, offset = _take(payload, offset, batch * dv * dk, "<f8")
        cases.append(BundleCase(
            case_id,
            keys.reshape(t, batch, dk), values.reshape(t, batch, dv),
            queries.reshape(t, batch, dk), alpha.reshape(t, batch),
            beta.reshape(t, batch), lengths, initial.reshape(batch, dv, dk),
            expected_outputs.reshape(t, batch, dv),
            expected_final.reshape(batch, dv, dk),
        ))
    if offset != len(payload):
        raise ValueError("trailing bytes in recurrence bundle")
    manifest = {
        "format": "KERNELWIKI-VOLTA-RECURRENCE-BUNDLE/1",
        "bundle_payload_sha256": payload_sha,
        "bundle_file_sha256": hashlib.sha256(binary).hexdigest(),
        "bundle_bytes": len(binary),
        "case_count": len(cases),
    }
    return cases, manifest


def write_bundle(path: str | Path, cases: list[BundleCase]) -> dict:
    binary, manifest = encode_bundle(cases)
    path = Path(path)
    path.write_bytes(binary)
    manifest["bundle_path"] = path.name
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
