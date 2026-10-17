"""CPU-only FP32 arithmetic scale probe; this is not CUDA evidence."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from bundle_io import ATOL, RTOL, decode_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cases, manifest = decode_bundle(args.bundle.read_bytes())
    results = []
    for case in cases:
        tmax, batch, dv, dk = case.shape
        # Exercise every short/decode sequence. For long prefill include all
        # short/ragged prefixes plus the two complete near-one trajectories.
        seqs = list(range(batch)) if tmax <= 33 else list(range(min(8, batch)))
        max_abs = 0.0
        max_scaled = 0.0
        checked = 0
        for b in seqs:
            state = case.initial_states[b].copy()
            for t in range(int(case.lengths[b])):
                a = np.float32(case.alpha[t, b])
                beta = np.float32(case.beta[t, b])
                key, value, query = case.keys[t, b], case.values[t, b], case.queries[t, b]
                out = np.empty(dv, dtype=np.float32)
                for row in range(dv):
                    decayed = np.empty(dk, dtype=np.float32)
                    dot = np.float32(0.0)
                    for j in range(dk):
                        decayed[j] = np.float32(a * state[row, j])
                        dot = np.float32(math.fma(float(decayed[j]), float(key[j]), float(dot)))
                    error = np.float32(value[row] - dot)
                    beta_error = np.float32(beta * error)
                    acc = np.float32(0.0)
                    for j in range(dk):
                        state[row, j] = np.float32(
                            math.fma(float(beta_error), float(key[j]), float(decayed[j]))
                        )
                        acc = np.float32(math.fma(float(state[row, j]), float(query[j]), float(acc)))
                    out[row] = acc
                ref = case.expected_outputs[t, b]
                abs_err = np.abs(out.astype(np.float64) - ref)
                scaled = abs_err / (ATOL + RTOL * np.abs(ref))
                max_abs = max(max_abs, float(np.max(abs_err)))
                max_scaled = max(max_scaled, float(np.max(scaled)))
                checked += dv
            final_err = np.abs(state.astype(np.float64) - case.expected_final_states[b])
            final_scaled = final_err / (ATOL + RTOL * np.abs(case.expected_final_states[b]))
            max_abs = max(max_abs, float(np.max(final_err)))
            max_scaled = max(max_scaled, float(np.max(final_scaled)))
            checked += state.size
        results.append({
            "case_id": case.case_id,
            "sequences_probed": len(seqs),
            "checked_values": checked,
            "max_abs_error": max_abs,
            "max_scaled_error": max_scaled,
            "within_predeclared_policy": max_scaled <= 1.0,
        })
    report = {
        "format": "KERNELWIKI-RECURRENCE-FP32-POLICY-PROBE/1",
        "evidence_kind": "cpu_only_arithmetic_sanity_check",
        "gpu_execution": False,
        "bundle_file_sha256": manifest["bundle_file_sha256"],
        "atol": ATOL,
        "rtol": RTOL,
        "arithmetic": "FP32 multiplies with fmaf-order round-to-FP32 recurrence; no CUDA instructions",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cases": results,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(x["within_predeclared_policy"] for x in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
