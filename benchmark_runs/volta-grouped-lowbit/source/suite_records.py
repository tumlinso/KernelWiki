"""Build bounded suite-level identities and pre-run CPU oracle wrappers.

The wrapper is a CPU preparation artifact only. Native GPU correctness remains
owned by the checked outputs emitted by kwv07_runner after each device run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from bundle import (
    KIND_GROUPED,
    KIND_W4A16,
    read_input,
    read_oracle,
    sha256_file,
)
from contract import (
    GroupedFixture,
    check_w4a16_fp32_simulation,
    gamma_bound,
    simulate_grouped_fp32,
    grouped_gated_admission_bound,
    silu64,
)

try:
    import numpy as np
except ImportError:  # Keep the CPU tooling dependency-free; this fallback is slower.
    np = None


SUITES = {
    "grouped": {
        "suite_id": "kwv07-grouped-gated",
        "family": "grouped_gated",
        "case_ids": ("tiny_balanced", "tiny_skewed", "moderate_balanced", "moderate_skewed",
                     "weighted_balanced", "weighted_skewed"),
        "operator": "Y[output_position] = silu32(X16[row] @ W_gate16[expert]) * (X16[row] @ W_up16[expert]); both projections accumulate in FP32",
        "numerical_policy": "FP16 represented inputs; FP32 projection/intermediate/SiLU/product/output; FP64 represented-input oracle; per-output bound abs(silu(G64))*eU + (abs(U64)+eU)*(1+1/e)*eG + 8*2^-24*(1+abs(Y64)+abs(U64)+abs(G64)), with eU/eG=gamma_K*sum_abs_products",
    },
    "w4a16": {
        "suite_id": "kwv07-w4a16",
        "family": "w4a16",
        "case_ids": ("low_m", "moderate_m", "weighted_low_m", "weighted_moderate_m"),
        "operator": "C[m,n] = sum_k A16[m,k] * round_fp16_rn(round_fp32(scale[floor(k/128),n] * signed_int4_q[k,n]))",
        "numerical_policy": "FP16 represented activations; signed two's-complement q in [-8,7], even row-major K*N code in low nibble; FP32 scale multiply followed by FP16 RN decode; FP32 accumulation/output; FP64 oracle; per-output gamma_K*sum(abs(A16*decoded_W16)) bound",
    },
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _grouped_precheck(fixture: GroupedFixture, reference: dict[str, Any]) -> tuple[bool, float, float]:
    """Check FP32 candidate arithmetic against the already hash-bound FP64 oracle."""
    if np is None:
        actual = simulate_grouped_fp32(fixture)
        errors, ratios = [], []
        for index, value in enumerate(actual):
            bound = grouped_gated_admission_bound(
                reference["up_reference"][index], reference["gate_reference"][index],
                reference["y64"][index], reference["sum_abs_up"][index],
                reference["sum_abs_gate"][index], fixture.k)
            error = abs(value - reference["y64"][index])
            errors.append(error)
            ratios.append(0.0 if bound == 0.0 and error == 0.0 else error / bound)
        return max(ratios, default=0.0) <= 1.0, max(errors, default=0.0), max(ratios, default=0.0)

    x = np.asarray(fixture.x16, dtype=np.float32).reshape(fixture.rows, fixture.k)
    up = np.asarray(fixture.up16, dtype=np.float32).reshape(fixture.experts, fixture.k, fixture.n)
    gate = np.asarray(fixture.gate16, dtype=np.float32).reshape(fixture.experts, fixture.k, fixture.n)
    actual = np.empty((len(fixture.assignments), fixture.n), dtype=np.float32)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for a in fixture.assignments:
            u = x[a.row_id] @ up[a.expert_id]
            g = x[a.row_id] @ gate[a.expert_id]
            silu = g / (np.float32(1.0) + np.exp(-g))
            actual[a.output_position] = silu * u
    y64 = np.asarray(reference["y64"], dtype=np.float64).reshape(actual.shape)
    up64 = np.asarray(reference["up_reference"], dtype=np.float64).reshape(actual.shape)
    gate64 = np.asarray(reference["gate_reference"], dtype=np.float64).reshape(actual.shape)
    sum_up = np.asarray(reference["sum_abs_up"], dtype=np.float64).reshape(actual.shape)
    sum_gate = np.asarray(reference["sum_abs_gate"], dtype=np.float64).reshape(actual.shape)
    gamma = (fixture.k * (2.0 ** -24)) / (1.0 - fixture.k * (2.0 ** -24))
    e_up, e_gate = gamma * sum_up, gamma * sum_gate
    # Use the exact real SiLU reference rather than infer its value from Y/U;
    # zero U makes the quotient undefined.
    silu64 = np.where(gate64 >= 0.0,
                      gate64 / (1.0 + np.exp(-gate64)),
                      gate64 * np.exp(gate64) / (1.0 + np.exp(gate64)))
    bounds = (np.abs(silu64) * e_up +
              (np.abs(up64) + e_up) * (1.0 + 1.0 / math.e) * e_gate +
              8.0 * (2.0 ** -24) * (1.0 + np.abs(y64) + np.abs(up64) + np.abs(gate64)))
    errors = np.abs(actual.astype(np.float64) - y64)
    ratios = np.divide(errors, bounds, out=np.full_like(errors, np.inf), where=bounds != 0.0)
    ratios[(bounds == 0.0) & (errors == 0.0)] = 0.0
    return bool(np.all(np.isfinite(actual)) and np.max(ratios, initial=0.0) <= 1.0), \
        float(np.max(errors, initial=0.0)), float(np.max(ratios, initial=0.0))


def _w4_precheck(parsed: dict[str, Any], reference: dict[str, Any]) -> tuple[bool, float, float]:
    """Vectorized CPU FP32 simulation; use FP64 output/sum files as fixed oracle."""
    if np is None:
        return check_w4a16_fp32_simulation(
            parsed["activation16"], parsed["m"], parsed["k"], parsed["n"],
            parsed["packed_codes"], parsed["scales"], reference["y64"],
            reference["sum_abs_products"], parsed["group_size"])
    k, n, m = parsed["k"], parsed["n"], parsed["m"]
    count = k * n
    packed = np.frombuffer(parsed["packed_codes"], dtype=np.uint8)
    codes = np.empty(count, dtype=np.int8)
    codes[0::2] = packed & 0x0f
    codes[0::2][codes[0::2] >= 8] -= 16
    if count > 1:
        codes[1::2] = packed[:count // 2] >> 4
        codes[1::2][codes[1::2] >= 8] -= 16
    scales = np.asarray(parsed["scales"], dtype=np.float32).reshape(-1, n)
    group_ids = np.arange(k, dtype=np.int64) // parsed["group_size"]
    scale_rows = scales[group_ids]
    decoded = (scale_rows * codes.reshape(k, n).astype(np.float32)).astype(np.float16).astype(np.float32)
    activation = np.asarray(parsed["activation16"], dtype=np.float32).reshape(m, k)
    actual = activation @ decoded
    y64 = np.asarray(reference["y64"], dtype=np.float64).reshape(m, n)
    sum_abs = np.asarray(reference["sum_abs_products"], dtype=np.float64).reshape(m, n)
    gamma = (k * (2.0 ** -24)) / (1.0 - k * (2.0 ** -24))
    bounds = gamma * sum_abs
    errors = np.abs(actual.astype(np.float64) - y64)
    ratios = np.divide(errors, bounds, out=np.full_like(errors, np.inf), where=bounds != 0.0)
    ratios[(bounds == 0.0) & (errors == 0.0)] = 0.0
    return bool(np.all(np.isfinite(actual)) and np.max(ratios, initial=0.0) <= 1.0), \
        float(np.max(errors, initial=0.0)), float(np.max(ratios, initial=0.0))


def _case_error_and_bound(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    parsed = read_input(root / record["input"]["path"])
    reference = read_oracle(root / record["oracle"]["path"])
    if parsed["kind"] != reference["kind"]:
        raise ValueError(f"{record['id']}: binary input/reference family mismatch")

    if parsed["kind"] == KIND_GROUPED:
        fixture = GroupedFixture(
            parsed["rows"], parsed["k"], parsed["n"], parsed["experts"],
            parsed["x16"], parsed["up16"], parsed["gate16"], parsed["assignments"],
        )
        passed, max_error, max_ratio = _grouped_precheck(fixture, reference)
        bounds = [
            grouped_gated_admission_bound(
                reference["up_reference"][i], reference["gate_reference"][i],
                reference["y64"][i], reference["sum_abs_up"][i],
                reference["sum_abs_gate"][i], parsed["k"],
            )
            for i in range(len(reference["y64"]))
        ]
        values = reference["y64"]
        checked_outputs = len(values)
        mode = "paired FP32 CPU accumulation/SiLU/product simulation; identity and per-output admission precheck only"
    elif parsed["kind"] == KIND_W4A16:
        passed, max_error, max_ratio = _w4_precheck(parsed, reference)
        bounds = [gamma_bound(parsed["k"], item) for item in reference["sum_abs_products"]]
        values = reference["y64"]
        checked_outputs = len(values)
        mode = "direct FP32 CPU dot simulation over exact decoded FP16 weights; identity and per-output admission precheck only"
    else:
        raise ValueError(f"{record['id']}: unsupported suite family")

    # check_* compares each output with its own bound. The aggregate scalar is
    # only the compact schema accepted by the evidence registry; the detailed
    # per-output GPU result will be emitted separately by the native runner.
    max_bound = max(bounds, default=0.0)
    reference_max_abs = max((abs(v) for v in values), default=0.0)
    if not passed:
        raise ValueError(f"{record['id']}: fixed CPU FP32 simulation failed (max ratio {max_ratio})")
    return {
        "case_id": record["id"],
        "checked_outputs": checked_outputs,
        "max_abs_error": max_error,
        "reference_max_abs": reference_max_abs,
        "atol": max_bound,
        "rtol": 0.0,
        "max_individual_bound_ratio": max_ratio,
        "cpu_check": mode,
        "result": "pass",
    }


def expected_suite_records(root: Path, manifest: dict[str, Any]) -> dict[str, bytes]:
    """Return deterministic suite artifact bytes without mutating the bundle."""
    root_manifest_path = root / "bundle-manifest.json"
    root_manifest_sha = sha256_file(root_manifest_path)
    records_by_id = {record["id"]: record for record in manifest["cases"]}
    artifacts: dict[str, bytes] = {}
    for key, spec in SUITES.items():
        selected = []
        checks = []
        for case_id in spec["case_ids"]:
            record = records_by_id[case_id]
            selected.append({
                "id": case_id,
                "input": record["input"],
                "oracle": record["oracle"],
                "contract": record["contract"],
            })
            checks.append(_case_error_and_bound(root, record))

        suite_manifest = {
            "format": "KERNELWIKI-KWV07-SUITE/1",
            "suite_id": spec["suite_id"],
            "family": spec["family"],
            "path_basis": "bundle root",
            "source_bundle_manifest": {"path": "bundle-manifest.json", "sha256": root_manifest_sha},
            "binary_format": manifest["binary_format"],
            "cases": selected,
        }
        suite_manifest_rel = f"suites/{key}.json"
        suite_manifest_data = _json_bytes(suite_manifest)
        suite_manifest_sha = _sha(suite_manifest_data)
        artifacts[suite_manifest_rel] = suite_manifest_data

        contract = {
            "format": "KERNELWIKI-KWV07-SUITE-CONTRACT/1",
            "case_id": spec["suite_id"],
            "case_signature": _sha(_json_bytes({
                "suite_id": spec["suite_id"],
                "bundle_sha256": root_manifest_sha,
                "cases": selected,
            })),
            "family": spec["family"],
            "path_basis": "bundle root",
            "suite_manifest": {"path": f"suites/{key}.json", "sha256": suite_manifest_sha},
            "case_ids": list(spec["case_ids"]),
            "input_references": [entry["input"] for entry in selected],
            "oracle_references": [entry["oracle"] for entry in selected],
            "numerical_contract": {
                "operator": spec["operator"],
                "policy": spec["numerical_policy"],
                "cpu_oracle": "FP64 CPU reference generated over exact represented binary inputs; exact binary references are SHA-bound and the declared FP32 arithmetic is prechecked before device execution",
                "gpu_check": "runner checks every output against the fixed per-output policy after each device invocation; this contract is not a GPU result",
            },
        }
        contract_rel = f"contracts/{key}-suite.json"
        artifacts[contract_rel] = _json_bytes(contract)

        oracle = {
            "format": "KERNELWIKI-CPU-ORACLE/1",
            "passed": True,
            "suite_id": spec["suite_id"],
            "validation_scope": "CPU-only represented-input/reference identity and fixed-policy FP32 simulation precheck; no GPU execution or GPU numerical result is asserted",
            "path_basis": "bundle root",
            "suite_manifest": {"path": f"suites/{key}.json", "sha256": suite_manifest_sha},
            "checks": checks,
        }
        oracle_rel = f"oracles/{key}-suite.json"
        artifacts[oracle_rel] = _json_bytes(oracle)
    return artifacts


def generate_suite_records(root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for relative, data in expected_suite_records(root, manifest).items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[relative] = _sha(data)
    integrity = {
        "format": "KERNELWIKI-KWV07-SUITE-FILES/1",
        "files": hashes,
    }
    integrity_data = _json_bytes(integrity)
    (root / "suite-files.json").write_bytes(integrity_data)
    hashes["suite-files.json"] = _sha(integrity_data)
    return hashes


def verify_suite_records(root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Check suite wrapper hashes and their exact links to root bundle records."""
    integrity_path = root / "suite-files.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("format") != "KERNELWIKI-KWV07-SUITE-FILES/1":
        raise ValueError("unknown KWV-07 suite-file integrity format")
    files = integrity.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("suite-file integrity manifest has no files")
    for relative, expected_sha in files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"suite wrapper identity/precheck mismatch: {relative}")

    root_manifest_sha = sha256_file(root / "bundle-manifest.json")
    root_cases = {entry["id"]: entry for entry in manifest["cases"]}
    for key, spec in SUITES.items():
        suite_path = f"suites/{key}.json"
        contract_path = f"contracts/{key}-suite.json"
        oracle_path = f"oracles/{key}-suite.json"
        suite = json.loads((root / suite_path).read_text(encoding="utf-8"))
        contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
        oracle = json.loads((root / oracle_path).read_text(encoding="utf-8"))
        if (suite.get("suite_id") != spec["suite_id"] or suite.get("family") != spec["family"] or
                suite.get("source_bundle_manifest") != {"path": "bundle-manifest.json", "sha256": root_manifest_sha}):
            raise ValueError(f"suite wrapper metadata does not match root bundle: {key}")
        entries = suite.get("cases")
        if not isinstance(entries, list) or [item.get("id") for item in entries] != list(spec["case_ids"]):
            raise ValueError(f"suite case membership mismatch: {key}")
        for entry in entries:
            source = root_cases[entry["id"]]
            for field in ("input", "oracle", "contract"):
                if entry.get(field) != source[field]:
                    raise ValueError(f"suite member {entry['id']} {field} binding mismatch")
        suite_sha = sha256_file(root / suite_path)
        if (contract.get("case_id") != spec["suite_id"] or
                contract.get("suite_manifest") != {"path": suite_path, "sha256": suite_sha} or
                contract.get("case_ids") != list(spec["case_ids"]) or
                not isinstance(contract.get("numerical_contract"), dict)):
            raise ValueError(f"suite contract does not bind the family manifest: {key}")
        if (oracle.get("format") != "KERNELWIKI-CPU-ORACLE/1" or oracle.get("passed") is not True or
                oracle.get("suite_manifest") != {"path": suite_path, "sha256": suite_sha} or
                "no GPU execution" not in oracle.get("validation_scope", "")):
            raise ValueError(f"suite oracle is not an explicit CPU-only precheck: {key}")
        checks = oracle.get("checks")
        if not isinstance(checks, list) or [check.get("case_id") for check in checks] != list(spec["case_ids"]):
            raise ValueError(f"suite oracle check membership mismatch: {key}")
        for check in checks:
            error, reference, atol, rtol = [check.get(field) for field in
                                           ("max_abs_error", "reference_max_abs", "atol", "rtol")]
            if (check.get("result") != "pass" or
                    any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value) or value < 0
                        for value in (error, reference, atol, rtol)) or
                    error > atol + rtol * reference):
                raise ValueError(f"suite CPU precheck is numerically malformed: {check.get('case_id')}")
    return {**files, "suite-files.json": sha256_file(integrity_path)}

