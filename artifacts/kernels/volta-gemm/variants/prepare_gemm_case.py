#!/usr/bin/env python3
"""Prepare immutable represented inputs and KWV-04 oracle/contract records.

This is a narrow case-preparation helper, not a benchmark or evidence runner.
The actual device result comparison is emitted by the native correctness mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SHAPES = {
    "large_dense": {"m": 1024, "n": 1024, "k": 1024, "seed": 0x70A001},
    "small_m": {"m": 16, "n": 4096, "k": 4096, "seed": 0x70A002},
    "ragged": {"m": 127, "n": 513, "k": 769, "seed": 0x70A003},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(SHAPES))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cpu-oracle", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    represented = args.output_dir / "inputs.bin"
    subprocess.run([str(args.cpu_oracle), "generate", args.case, str(represented)], check=True)
    reference = args.output_dir / "reference.f64le"
    subprocess.run([str(args.cpu_oracle), "generate-reference", args.case,
                    str(represented), str(reference)], check=True)
    input_hash = sha256(represented)
    reference_hash = sha256(reference)
    numerical_contract = {
        "operator": "C[m,n] = sum_k A16[m,k] * B16[k,n]",
        "layout": "row-major A[M,K], B[K,N], C[M,N]",
        "represented_inputs": "serialized IEEE binary16 bit patterns; all routes consume these exact bits",
        "reference_output": "immutable per-case binary64 C and sum_abs_products generated from represented_inputs on CPU before any device run",
        "accumulator": "FP32; alpha=1; beta=0",
        "core_output": "FP32",
        "epilogue": "optional FP32 bias[n] followed by ReLU; no intermediate rounding beyond FP32 C",
        "core_admission_tolerance": "gamma_K * sum_k(abs(double(A16[m,k]) * double(B16[k,n])))",
        "epilogue_admission_tolerance": "gamma_(K+1) * (sum_abs_products + abs(bias[n]))",
        "unit_roundoff": "2^-24",
        "tolerance_interpretation": "gamma-scaled thresholds are a predeclared per-output FP32 admission policy checked against every output; they are not a universal proved hardware error bound because PTX MMA accumulation order and rounding are unspecified",
        "exceptional_values": "inputs and bias finite; no NaN/Inf; generated range avoids overflow and subnormal inputs",
    }
    case_contract = {
        "case_id": args.case,
        "shape": SHAPES[args.case],
        "case_signature": canonical_sha({"case": args.case, "shape": SHAPES[args.case],
                                          "input_sha256": input_hash,
                                          "reference_sha256": reference_hash,
                                          "numerical_contract": numerical_contract}),
        "numerical_contract": numerical_contract,
        "represented_input": {"path": "inputs.bin", "sha256": input_hash},
        "reference_output": {"path": "reference.f64le", "sha256": reference_hash,
                             "format": "KWVREF01: little-endian dimensions/seed, row-major FP64 C, row-major FP64 sum_abs_products"},
    }
    (args.output_dir / "contract.json").write_text(
        json.dumps(case_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    oracle = json.loads(subprocess.check_output(
        [str(args.cpu_oracle), "oracle-self-test", args.case], text=True))
    oracle.update({
        "represented_input": {"path": "inputs.bin", "sha256": input_hash},
        "reference_output": {"path": "reference.f64le", "sha256": reference_hash,
                             "format": "KWVREF01: exact generated case reference output and per-output absolute-product sum"},
        "expected_reference_generation": {
            "source": "artifacts/kernels/volta-gemm/variants/gemm_contract.hpp: compute_oracle",
            "inputs": "read serialized binary16 values, convert each exactly to binary64, accumulate each output in binary64",
            "outputs": "the hashed KWVREF01 file stores C64 and sum_abs_products for each row-major output; controller records bind this exact file as an input",
            "device_comparison": "each native correctness run emits its own per-output error, bound, failures, and exit status",
        },
    })
    (args.output_dir / "cpu-oracle.json").write_text(
        json.dumps(oracle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case_id": args.case, "input": str(represented),
                      "input_sha256": input_hash,
                      "reference": str(reference), "reference_sha256": reference_hash,
                      "contract": str(args.output_dir / "contract.json"),
                      "oracle": str(args.output_dir / "cpu-oracle.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
