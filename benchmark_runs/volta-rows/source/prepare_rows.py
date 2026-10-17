#!/usr/bin/env python3
"""Write deterministic KWV-05 represented-input and FP64 reference suites."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from row_oracle import (ATOL, RTOL, Case, compute_reference, encode_bundle,
                        encode_reference, oracle_record)


def _sample(seed: int, index: int, bound: int, scale: int) -> float:
    # Integer-only deterministic generator; no runtime library RNG or device data.
    bits = (index * 1664525 + seed * 1013904223) & 0xFFFFFFFF
    quantized = int(bits % (2 * bound + 1)) - bound
    return struct.unpack("<f", struct.pack("<f", quantized / scale))[0]


def _make_case(operation: str, case_id: str, rows: int, cols: int,
               *, masked: bool = False, misaligned: bool = False,
               zero_values: bool = False) -> Case:
    x_offset = 1 if misaligned and rows else 0
    y_offset = 3 if misaligned and rows else 0
    x_stride = cols + (1 if misaligned else (-cols) % 4)
    y_stride = cols + (2 if misaligned else (-cols) % 4)
    mask_stride = cols + 3
    mask_offset = 2 if misaligned and masked and rows else 0
    x_count = (x_offset + (rows - 1) * x_stride + cols) if rows else 0
    x = [0.0] * x_count
    for row in range(rows):
        for col in range(cols):
            value = 0.0 if zero_values else _sample(70_005 + cols + row, row * cols + col, 4096,
                                                     256 if operation == "softmax" else 64)
            x[x_offset + row * x_stride + col] = value
    mask: list[int] = []
    if masked:
        mask_count = mask_offset + (rows - 1) * mask_stride + cols if rows else 0
        mask = [0] * mask_count
        for row in range(rows):
            for col in range(cols):
                mask[mask_offset + row * mask_stride + col] = int((row * 3 + col) % 7 != 0)
        if rows:
            # Include a fully masked row, which must emit exact zeros.
            for col in range(cols):
                mask[mask_offset + (rows - 1) * mask_stride + col] = 0
    gamma_offset = 1 if operation == "rmsnorm" and misaligned else 0
    gamma = []
    epsilon = 0.0
    if operation == "rmsnorm":
        gamma = [0.0] * (gamma_offset + cols)
        for col in range(cols):
            gamma[gamma_offset + col] = _sample(90_005 + cols, col, 2048, 512)
        epsilon = 1.0e-4
    case = Case(case_id, operation, rows, cols, x_stride, y_stride, mask_stride,
                x_offset, y_offset, mask_offset, gamma_offset, epsilon,
                x, mask, gamma, masked)
    return case


def suites() -> dict[str, list[Case]]:
    widths = (31, 32, 33, 128, 768, 4096, 8193)
    softmax = [_make_case("softmax", f"softmax_n{width}", 3, width)
               for width in widths]
    softmax.extend([
        _make_case("softmax", "softmax_mask_tail_stride", 5, 33,
                   masked=True, misaligned=True),
        _make_case("softmax", "softmax_zero_rows", 0, 33),
    ])
    rmsnorm = [_make_case("rmsnorm", f"rmsnorm_n{width}", 3, width)
               for width in widths]
    rmsnorm.extend([
        _make_case("rmsnorm", "rmsnorm_zero_row", 1, 33, zero_values=True),
        _make_case("rmsnorm", "rmsnorm_zero_rows", 0, 33),
        _make_case("rmsnorm", "rmsnorm_n33_misaligned", 5, 33, misaligned=True),
    ])
    return {"softmax": softmax, "rmsnorm": rmsnorm}


def performance_cases() -> list[Case]:
    return [
        _make_case("softmax", "softmax_wide_single_row", 1, 4096),
        _make_case("softmax", "softmax_wide_streaming", 512, 4096),
        _make_case("rmsnorm", "rmsnorm_vec4_wide_streaming", 512, 4096),
    ]


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def route_manifest(case: Case) -> dict[str, object]:
    """Describe exact native implementation paths for both fixed routes."""
    if case.rows == 0:
        baseline_path = candidate_path = "no_launch"
    elif case.operation == "softmax":
        baseline_path, candidate_path = "softmax_warp", "softmax_cta"
    else:
        baseline_path = "rmsnorm_cta_scalar"
        aligned = (case.cols >= 4 and case.x_offset % 4 == 0 and
                   case.gamma_offset % 4 == 0 and case.y_offset % 4 == 0 and
                   case.x_row_stride % 4 == 0 and case.y_row_stride % 4 == 0)
        candidate_path = "vec4" if aligned else "scalar_fallback"
    return {
        "case_id": case.case_id,
        "operation": case.operation,
        "rows": case.rows,
        "cols": case.cols,
        "expected_outputs": case.logical_count,
        "baseline_path": baseline_path,
        "candidate_path": candidate_path,
        "check_identity": f"logical-row-major:{case.case_id}",
    }


def write_combined_suite(cases_by_op: dict[str, list[Case]], output_dir: Path,
                         measured_cases: list[Case]) -> dict[str, str]:
    cases = [case for group in cases_by_op.values() for case in group] + measured_cases
    output_dir.mkdir(parents=True, exist_ok=True)
    input_bytes = encode_bundle(cases)
    reference_bytes = encode_reference([(case.case_id, compute_reference(case)) for case in cases])
    input_path = output_dir / "inputs.bin"
    reference_path = output_dir / "reference.f64le"
    input_path.write_bytes(input_bytes)
    reference_path.write_bytes(reference_bytes)
    oracle_path = output_dir / "cpu-oracle.json"
    oracle = oracle_record(input_bytes, reference_bytes)
    oracle_path.write_text(json.dumps(oracle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    numerical_contract = {
            "softmax": {
                "definition": "masked row-stable softmax; all-masked rows return zero",
                "domain": "logical finite FP32 x in [-16,16]; masks are bytes 0/1",
            },
            "rmsnorm": {
                "definition": "plain gamma*x*rsqrt(epsilon+sum(x*x)/N); no mask, residual, or bias",
                "domain": "logical finite FP32 abs(x)<=64, abs(gamma)<=4, epsilon in [1e-5,1e-2]",
                "maximum_width": 8193,
                "sum_of_squares_upper_bound": 4096 * 8193,
            },
            "common": {
                "storage": "exact little-endian IEEE FP32 represented input; independent Python binary64 oracle",
                "tolerance_atol": ATOL, "tolerance_rtol": RTOL,
                "comparison": "every logical output: abs(actual-reference) <= atol + rtol*abs(reference)",
                "layout": "1<=N<=8193; zero rows allowed and skip launch; row strides >=N; independent offsets; no aliasing",
                "padding": "ignored input padding; output padding must remain unchanged",
            },
    }
    signature_payload = {
        "suite_id": "rows_correctness_suite",
        "case_ids": [case.case_id for case in cases],
        "cases": [route_manifest(case) for case in cases],
        "represented_input_sha256": sha256(input_bytes),
        "reference_output_sha256": sha256(reference_bytes),
        "numerical_contract": numerical_contract,
    }
    case_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    contract = {
        "format": "KERNELWIKI-ROW-CONTRACT/1",
        "case_id": "rows",
        "case_signature": case_signature,
        "suite_id": "rows_correctness_suite",
        "operators": ["softmax", "rmsnorm"],
        "case_ids": [case.case_id for case in cases],
        "cases": [route_manifest(case) for case in cases],
        "correctness_case_ids": [case.case_id for case in cases],
        "selected_timing_case_ids": [case.case_id for case in measured_cases],
        "represented_input": {"path": input_path.name, "sha256": sha256(input_bytes)},
        "reference_output": {"path": reference_path.name, "sha256": sha256(reference_bytes),
                            "format": "KWVRREF1: per-case logical row-major FP64 outputs"},
        "cpu_oracle": {"path": oracle_path.name, "sha256": sha256(oracle_path.read_bytes())},
        "numerical_contract": numerical_contract,
    }
    contract_path = output_dir / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"suite": "rows", "case_count": len(cases),
            "input_sha256": sha256(input_bytes), "reference_sha256": sha256(reference_bytes),
            "oracle_sha256": sha256(oracle_path.read_bytes()),
            "contract_sha256": sha256(contract_path.read_bytes())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    case_suites = suites()
    result = write_combined_suite(case_suites, args.output_dir, performance_cases())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
