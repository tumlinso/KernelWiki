#!/usr/bin/env python3
"""Recompute the smoke's binary64 references from its exact FP16 input file.

This checks the independent CPU reference magnitudes recorded in
benchmark_runs/volta-environment/cpu-oracle.json. The native controller
receipts separately contain the GPU comparison results; this helper does not
claim to re-execute or independently validate a GPU.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(value: Fraction) -> float:
    """Round an exact rational to IEEE binary32, then return it as Python float."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def recompute(input_path: Path) -> dict:
    payload = input_path.read_bytes()
    if len(payload) != 2 * 16 * 16 * 2:
        raise ValueError("input must contain exactly two 16x16 FP16 matrices")
    values = struct.unpack("<512e", payload)
    left = values[:256]
    right = values[256:]
    left_exact = [Fraction.from_float(float(value)) for value in left]
    right_exact = [Fraction.from_float(float(value)) for value in right]

    fma_errors = []
    for index in range(64):
        a = left_exact[index]
        b = right_exact[index]
        c = Fraction((index * 5 % 11) - 5, 32)
        exact = a * b + c
        fma_errors.append(abs(f32(exact) - float(exact)))

    gemm_errors = []
    gemm_reference_abs = []
    for row in range(16):
        for col in range(16):
            exact = sum(
                (left_exact[row * 16 + k] * right_exact[k * 16 + col]
                 for k in range(16)),
                Fraction(0),
            )
            gemm_reference_abs.append(abs(float(exact)))
            gemm_errors.append(abs(f32(exact) - float(exact)))

    return {
        "input_sha256": digest(input_path),
        "inline-ptx-fma-rn-f32": {
            "elements": 64,
            "reference_max_abs": max(abs(float(
                left_exact[i] * right_exact[i] + Fraction((i * 5 % 11) - 5, 32)
            )) for i in range(64)),
            "binary32_rounding_max_abs_error": max(fma_errors),
        },
        "wmma-f16-input-f32-accum": {
            "m": 16,
            "n": 16,
            "k": 16,
            "reference_max_abs": max(gemm_reference_abs),
            "binary32_rounding_max_abs_error": max(gemm_errors),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True,
                        help="recorded CPU-oracle JSON to compare with the recomputation")
    args = parser.parse_args(argv)
    try:
        observed = recompute(args.input)
        oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
        expected_input = oracle.get("input", {}).get("sha256")
        if expected_input != observed["input_sha256"]:
            raise ValueError("oracle input SHA-256 does not match the represented input")
        checks = {item.get("name"): item for item in oracle.get("checks", [])
                  if isinstance(item, dict)}
        for name, result in observed.items():
            if name == "input_sha256":
                continue
            check = checks.get(name)
            if not isinstance(check, dict):
                raise ValueError(f"oracle record lacks check {name}")
            for field in ("reference_max_abs",):
                recorded = check.get(field)
                if (isinstance(recorded, bool) or not isinstance(recorded, (int, float)) or
                        not math.isclose(float(recorded), result[field], rel_tol=0, abs_tol=1e-15)):
                    raise ValueError(f"oracle {name} {field} differs from recomputation")
        observed["cpu_recomputation_passed"] = True
        observed["gpu_comparison"] = {
            "controller_gate_evidence_id": oracle.get("controller_gate_evidence_id"),
            "reported_max_abs_errors": {
                name: checks[name].get("max_abs_error")
                for name in ("inline-ptx-fma-rn-f32", "wmma-f16-input-f32-accum")
            },
            "independently_verified_by_this_helper": False,
        }
        print(json.dumps(observed, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"oracle verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
