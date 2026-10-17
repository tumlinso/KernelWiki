#!/usr/bin/env python3
"""Run bounded Volta GEMM alternatives inside one controller foreground lease."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "benchmark_runs/volta-gemm/bin/gemm_runner"
CASES = {"large_dense", "small_m", "ragged"}

CORE = [
    ("cublas_core", ["--impl", "cublas"]),
    ("ptx_stage2_core", ["--impl", "ptx", "--stages", "2"]),
    ("ptx_stage1_core", ["--impl", "ptx", "--stages", "1"]),
    ("cutlass_core", ["--impl", "cutlass"]),
]
EPILOGUE = [
    ("cublas_unfused_bias_relu", ["--impl", "cublas", "--bias-relu"]),
    ("ptx_fused_bias_relu", ["--impl", "ptx", "--stages", "2", "--bias-relu"]),
]


def run_once(case: str, variant: str, impl_args: list[str], phase: str) -> dict:
    case_dir = ROOT / "benchmark_runs/volta-gemm/cases" / case
    command = [
        str(RUNNER), "--case", case,
        "--input", str(case_dir / "inputs.bin"),
        *impl_args,
        "--mode", "correctness" if phase == "correctness" else "time",
    ]
    if phase == "correctness":
        command.extend(["--reference", str(case_dir / "reference.f64le")])
    else:
        # One native event sample per invocation. The outer loop gathers the
        # independent matched samples while the controller lease is held.
        command.extend(["--warmups", "1", "--repetitions", "1"])
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    result = {
        "variant": variant,
        "returncode": completed.returncode,
        "stderr": completed.stderr,
    }
    try:
        result["native"] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result["stdout"] = completed.stdout
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--phase", choices=("correctness", "timing"), required=True)
    args = parser.parse_args()

    plans = CORE + EPILOGUE
    results: dict[str, dict] = {}
    if args.phase == "correctness":
        for name, impl_args in plans:
            results[name] = run_once(args.case, name, impl_args, args.phase)
    else:
        samples: dict[str, list[float]] = {name: [] for name, _ in plans}
        native_records: dict[str, list[dict]] = {name: [] for name, _ in plans}
        failures: dict[str, list[dict]] = {name: [] for name, _ in plans}
        order = list(range(len(plans)))
        for repetition in range(11):
            offset = repetition % len(order)
            round_order = order[offset:] + order[:offset]
            for idx in round_order:
                name, impl_args = plans[idx]
                result = run_once(args.case, name, impl_args, args.phase)
                native = result.get("native", {})
                values = native.get("samples_us", [])
                native_records[name].append({
                    "repetition": repetition,
                    "returncode": result["returncode"],
                    "native": native,
                    "stderr": result.get("stderr", ""),
                })
                if result["returncode"] != 0 or len(values) != 1 or not math.isfinite(values[0]):
                    failures[name].append({"repetition": repetition, **result})
                    continue
                samples[name].append(float(values[0]))
        for name, _ in plans:
            values = samples[name]
            results[name] = {
                "samples_us": values,
                "sample_count": len(values),
                "median_us": statistics.median(values) if values else None,
                "minimum_us": min(values) if values else None,
                "maximum_us": max(values) if values else None,
                "complete": len(values) == 11,
                "native_records": native_records[name],
                "failures": failures[name],
            }

    complete = all(
        item.get("returncode", 0) == 0 and
        item.get("native", {}).get("passed") is True
        for item in results.values()
    ) if args.phase == "correctness" else all(
        item.get("complete") is True for item in results.values()
    )
    print(json.dumps({
        "format": "KERNELWIKI-GEMM-SUPPLEMENT/1",
        "case_id": args.case,
        "phase": args.phase,
        "device_pairing": "all variants in this record use the controller-selected GPU UUID",
        "results": results,
        "complete": complete,
    }, sort_keys=True, separators=(",", ":")))
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
