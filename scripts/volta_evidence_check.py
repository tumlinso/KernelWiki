#!/usr/bin/env python3
"""Fail-closed checks for source-bound, controller-backed Volta comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

FORMAT = "KERNELWIKI-VOLTA-EVIDENCE/1"
REQUIRED_RUNS = {
    "correctness_baseline": ("foreground-baseline", "baseline"),
    "correctness_candidate": ("foreground-baseline", "candidate"),
    "sanitizer_candidate": ("foreground-compute-sanitizer", "candidate"),
    "baseline_timing": ("foreground-baseline", "baseline"),
    "candidate_timing": ("foreground-baseline", "candidate"),
    "profile_candidate": ("foreground-ncu", "candidate"),
}
CASE_PATHS = {
    "environment": "benchmark_runs/volta-environment/summary.json",
    "gemm": "benchmark_runs/volta-gemm/summary.json",
    "rows": "benchmark_runs/volta-rows/summary.json",
    "attention": "benchmark_runs/volta-attention/summary.json",
    "grouped-lowbit": "benchmark_runs/volta-grouped-lowbit/summary.json",
    "recurrence": "benchmark_runs/volta-recurrence/summary.json",
}


class EvidenceError(ValueError):
    pass


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON {path}: {exc}") from exc


def confined(base: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label}: path is missing")
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()):
        raise EvidenceError(f"{label}: path escapes the evidence directory")
    if not path.is_file():
        raise EvidenceError(f"{label}: file is missing: {value}")
    return path


def checked_file(base: Path, record: Any, label: str) -> Path:
    if not isinstance(record, dict):
        raise EvidenceError(f"{label}: expected path and sha256")
    path = confined(base, record.get("path"), label)
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise EvidenceError(f"{label}: SHA-256 is missing or malformed")
    actual = digest(path)
    if actual != expected:
        raise EvidenceError(f"{label}: SHA-256 mismatch")
    return path


def repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / ".git").exists() or (parent / "data" / "schemas.yaml").is_file():
            return parent.resolve()
    raise EvidenceError("cannot locate repository root")


def absolute_repo_file(root: Path, raw_path: str, label: str) -> Path:
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise EvidenceError(f"{label}: file is missing or outside the repository")
    return path


def sha_binding(root: Path, record: Any, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise EvidenceError(f"{label}: expected path and sha256")
    raw_path, expected = record.get("path"), record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise EvidenceError(f"{label}: path or sha256 is missing")
    path = absolute_repo_file(root, raw_path, label)
    if digest(path) != expected:
        raise EvidenceError(f"{label}: SHA-256 mismatch")
    return path, expected


def _input_bindings(summary: dict[str, Any]) -> dict[str, str]:
    provenance = summary.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, dict) else None
    result: dict[str, str] = {}
    if isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str):
                result[str(Path(item["path"]).resolve())] = item["sha256"]
    return result


def _binary_binding(summary: dict[str, Any], root: Path, expected_path: Path,
                    expected_sha: str) -> None:
    binaries = summary.get("binaries")
    if not isinstance(binaries, list):
        raise EvidenceError("controller receipt does not list binary_paths")
    target = str(expected_path.resolve())
    if not any(isinstance(item, dict) and item.get("path") == target and
               item.get("sha256") == expected_sha for item in binaries):
        raise EvidenceError(f"controller receipt is not bound to binary {expected_path}")
    provenance = summary.get("provenance")
    binary = provenance.get("binary") if isinstance(provenance, dict) else None
    if (not isinstance(binary, dict) or binary.get("sha256") != expected_sha or
            not isinstance(binary.get("path"), str) or
            Path(binary["path"]).resolve() != expected_path.resolve()):
        raise EvidenceError("controller argv binary hash differs from the declared implementation")


def _controller_run(base: Path, root: Path, run: Any, label: str, expected_kind: str,
                    implementation: dict[str, Any], contract: Path, contract_sha: str,
                    inputs_path: Path, inputs_sha: str, oracle_path: Path,
                    oracle_sha: str, gpu_uuid: str,
                    evidence_ids: set[str]) -> dict[str, Any]:
    if not isinstance(run, dict):
        raise EvidenceError(f"{label}: controller receipt is missing")
    execution_path = checked_file(base, run.get("execution"), f"{label} execution")
    evidence_path = checked_file(base, run.get("evidence"), f"{label} evidence")
    # The durable lease is stored in the repo runtime tree; accept a repository-relative path.
    lease_record = run.get("lease")
    if not isinstance(lease_record, dict):
        raise EvidenceError(f"{label}: lease receipt is missing")
    lease_path = (root / str(lease_record.get("path", ""))).resolve()
    if not lease_path.is_relative_to(root) or not lease_path.is_file():
        raise EvidenceError(f"{label}: controller lease receipt is unavailable")
    if digest(lease_path) != lease_record.get("sha256"):
        raise EvidenceError(f"{label}: lease SHA-256 mismatch")

    execution = read_json(execution_path)
    exported = read_json(evidence_path)
    evidence = exported.get("evidence") if isinstance(exported, dict) else None
    if not isinstance(evidence, dict):
        raise EvidenceError(f"{label}: controller evidence export has no evidence object")
    evidence_id = evidence.get("id")
    if not isinstance(evidence_id, str) or not evidence_id or evidence_id != execution.get("evidence_id"):
        raise EvidenceError(f"{label}: execution and stored evidence IDs do not match")
    if evidence_id in evidence_ids:
        raise EvidenceError(f"{label}: evidence ID was reused for another measurement role")
    evidence_ids.add(evidence_id)
    if execution.get("ok") is not True or execution.get("status") != "succeeded":
        raise EvidenceError(f"{label}: controller execution did not succeed")
    if evidence.get("kind") != expected_kind or evidence.get("status") != "succeeded":
        raise EvidenceError(f"{label}: wrong or failed controller evidence kind")
    execution_lease = execution.get("lease_receipt")
    if (not isinstance(execution_lease, str) or not execution_lease or
            Path(execution_lease).resolve() != lease_path):
        raise EvidenceError(f"{label}: lease path differs from execution receipt")
    if execution.get("returncode", 0) != 0:
        raise EvidenceError(f"{label}: controller process return code was not zero")
    summary = evidence.get("summary")
    if not isinstance(summary, dict) or summary.get("valid") is not True or summary.get("status") != "succeeded":
        raise EvidenceError(f"{label}: stored controller result is invalid or failed")
    record = summary.get("record")
    if isinstance(record, dict) and record.get("returncode") != 0:
        raise EvidenceError(f"{label}: command return code was not zero")
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise EvidenceError(f"{label}: controller provenance is missing")
    toolchain = summary.get("toolchain")
    if (not isinstance(toolchain, dict) or toolchain.get("version") != "12.9" or
            not toolchain.get("nvcc") or not toolchain.get("sanitizer")):
        raise EvidenceError(f"{label}: actual CUDA 12.9 compiler/sanitizer identity is missing")
    source = provenance.get("source")
    if not isinstance(source, dict) or Path(str(source.get("project_root", ""))).resolve() != root:
        raise EvidenceError(f"{label}: controller source snapshot is unbound to this repository")
    expected_inputs = {
        str(contract.resolve()): contract_sha,
        str(inputs_path.resolve()): inputs_sha,
        str(oracle_path.resolve()): oracle_sha,
        str(implementation["source_path"].resolve()): implementation["source_sha"],
    }
    actual_inputs = _input_bindings(summary)
    if any(actual_inputs.get(path) != value for path, value in expected_inputs.items()):
        raise EvidenceError(f"{label}: controller inputs do not bind the contract, represented data, and source")
    _binary_binding(summary, root, implementation["executed_path"], implementation["binary_sha"])

    gpus = provenance.get("gpus")
    if not isinstance(gpus, list) or not any(
        isinstance(gpu, dict) and gpu.get("uuid") == gpu_uuid and
        str(gpu.get("compute_capability")) == "7.0" for gpu in gpus
    ):
        raise EvidenceError(f"{label}: controller evidence is not bound to the declared SM70 GPU")
    quiescence = (summary.get("resource_samples") or {}).get("quiescence")
    if not isinstance(quiescence, dict) or quiescence.get("uncontaminated") is not True:
        raise EvidenceError(f"{label}: controller quiescence receipt is absent or contaminated")
    lease = read_json(lease_path)
    expected_resource = f"accelerator:{gpu_uuid}"
    if (lease.get("format") != "CUDA-FOREGROUND-LEASE/1" or
            lease.get("project_root") != str(root) or
            expected_resource not in lease.get("resource_ids", [])):
        raise EvidenceError(f"{label}: foreground lease is not bound to the declared GPU and repository")
    return summary


def _timing_samples(summary: dict[str, Any], label: str) -> list[float]:
    benchmark = summary.get("benchmark")
    statistics = benchmark.get("statistics") if isinstance(benchmark, dict) else None
    values = statistics.get("values") if isinstance(statistics, dict) else None
    if not isinstance(values, list) or not values:
        raise EvidenceError(f"{label}: unprofiled timing samples are missing")
    samples: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise EvidenceError(f"{label}: timing sample is non-finite or non-positive")
        samples.append(float(value))
    return samples


def _partial_memcheck_has_application_pass(summary: dict[str, Any], gpu_uuid: str) -> bool:
    """Accept only the known partial memcheck classification with a raw target PASS."""
    if (summary.get("tool") != "compute-sanitizer" or
            summary.get("tool_detail") != "memcheck" or
            type(summary.get("exit_status")) is not int or
            summary.get("exit_status") != 0 or summary.get("conclusive") is not False):
        return False
    expected_uuid = gpu_uuid.removeprefix("GPU-").replace("-", "").lower()
    lines = summary.get("stdout_tail")
    if not isinstance(lines, list):
        return False
    for line in lines:
        if not isinstance(line, str):
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_uuid = result.get("gpu_uuid") if isinstance(result, dict) else None
        if (isinstance(result, dict) and result.get("status") == "passed" and
                str(result.get("compute_capability")) == "7.0" and
                isinstance(raw_uuid, str) and raw_uuid.lower() == expected_uuid):
            return True
    return False


def _check_case(base: Path, root: Path, case: Any, evidence_ids: set[str]) -> None:
    if not isinstance(case, dict) or not isinstance(case.get("id"), str):
        raise EvidenceError("case is malformed or lacks id")
    disposition = case.get("disposition")
    if disposition == "not_run":
        raise EvidenceError(f"{case['id']}: not_run is incomplete, not evaluated_not_promoted")
    if disposition not in {"retained", "evaluated_not_promoted"}:
        raise EvidenceError(f"{case['id']}: disposition is not a completed measurement")
    contract = checked_file(base, case.get("contract"), f"{case['id']} contract")
    contract_sha = digest(contract)
    contract_data = read_json(contract)
    if contract_data.get("case_id") != case["id"] or not contract_data.get("case_signature") or not contract_data.get("numerical_contract"):
        raise EvidenceError(f"{case['id']}: contract lacks case signature or numerical policy")
    input_path, input_sha = sha_binding(root, case.get("represented_input"), f"{case['id']} represented input")
    if case.get("represented_input", {}).get("sha256") != input_sha:
        raise EvidenceError(f"{case['id']}: represented input binding mismatch")

    implementations = case.get("implementations")
    if not isinstance(implementations, dict):
        raise EvidenceError(f"{case['id']}: implementations are missing")
    bound_impls: dict[str, dict[str, Any]] = {}
    for name in ("baseline", "candidate"):
        entry = implementations.get(name)
        if not isinstance(entry, dict):
            raise EvidenceError(f"{case['id']}: {name} implementation is missing")
        source_path, source_sha = sha_binding(root, entry.get("source"), f"{case['id']} {name} source")
        binary_path, binary_sha = sha_binding(root, entry.get("binary"), f"{case['id']} {name} binary")
        binary_record = entry.get("binary")
        default_executed = binary_record.get("path") if isinstance(binary_record, dict) else None
        executed_raw = entry.get("executed_path", default_executed)
        executed_path = absolute_repo_file(root, executed_raw, f"{case['id']} {name} executed binary")
        bound_impls[name] = {"source_path": source_path, "source_sha": source_sha,
                             "binary_path": binary_path, "executed_path": executed_path,
                             "binary_sha": binary_sha}

    libraries = case.get("libraries")
    if not isinstance(libraries, list):
        raise EvidenceError(f"{case['id']}: linked library identity must be an explicit list (empty if none)")
    bound_libraries = []
    for item in libraries:
        if (not isinstance(item, dict) or
                item.get("implementation") not in {"baseline", "candidate"} or
                not all(item.get(key) for key in ("name", "version", "path", "sha256"))):
            raise EvidenceError(f"{case['id']}: linked library entry lacks identity/version/path/hash")
        raw_library_path = Path(str(item["path"]))
        library_path = (raw_library_path if raw_library_path.is_absolute() else root / raw_library_path).resolve()
        if not library_path.is_file() or digest(library_path) != item["sha256"]:
            raise EvidenceError(f"{case['id']}: linked library path/hash is unavailable or mismatched")
        bound_libraries.append((item, library_path))

    oracle_path = checked_file(base, case.get("oracle"), f"{case['id']} CPU oracle")
    oracle_sha = digest(oracle_path)
    oracle = read_json(oracle_path)
    if oracle.get("format") != "KERNELWIKI-CPU-ORACLE/1" or oracle.get("passed") is not True:
        raise EvidenceError(f"{case['id']}: independent CPU oracle did not pass")
    checks = oracle.get("checks")
    if not isinstance(checks, list) or not checks:
        raise EvidenceError(f"{case['id']}: CPU oracle has no numeric checks")
    for check in checks:
        if not isinstance(check, dict):
            raise EvidenceError(f"{case['id']}: malformed CPU oracle comparison")
        raw_values = [check.get(key) for key in ("max_abs_error", "reference_max_abs", "atol", "rtol")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_values):
            raise EvidenceError(f"{case['id']}: CPU oracle comparison lacks numeric error/tolerance")
        error, reference, atol, rtol = (float(value) for value in raw_values)
        if (not all(math.isfinite(v) and v >= 0 for v in (error, reference, atol, rtol)) or
                error > atol + rtol * reference):
            raise EvidenceError(f"{case['id']}: CPU oracle numerical contract failed")

    runs = case.get("runs")
    if not isinstance(runs, dict):
        raise EvidenceError(f"{case['id']}: controller runs are missing")
    summaries: dict[str, dict[str, Any]] = {}
    for role, (kind, implementation_name) in REQUIRED_RUNS.items():
        summaries[role] = _controller_run(
            base, root, runs.get(role), f"{case['id']} {role}", kind,
            bound_impls[implementation_name], contract, contract_sha,
            input_path, input_sha, oracle_path, oracle_sha,
            str(case.get("gpu_uuid", "")), evidence_ids,
        )
    library_roles = {
        "baseline": ("correctness_baseline", "baseline_timing"),
        "candidate": ("correctness_candidate", "sanitizer_candidate", "candidate_timing", "profile_candidate"),
    }
    for library, library_path in bound_libraries:
        library_sha = str(library["sha256"])
        for role in library_roles[library["implementation"]]:
            if _input_bindings(summaries[role]).get(str(library_path)) != library_sha:
                raise EvidenceError(
                    f"{case['id']} {role}: controller inputs do not bind declared library {library['name']}"
                )
    _timing_samples(summaries["baseline_timing"], f"{case['id']} baseline timing")
    _timing_samples(summaries["candidate_timing"], f"{case['id']} candidate timing")

    sanitizer_record = runs["sanitizer_candidate"].get("sanitizer_summary")
    sanitizer_path = checked_file(base, sanitizer_record, f"{case['id']} sanitizer summary")
    sanitizer = read_json(sanitizer_path)
    sanitizer_status = sanitizer.get("status")
    sanitizer_passed = sanitizer_status in {"passed", "success", "succeeded"}
    partial_memcheck_passed = (
        sanitizer_status == "partial" and
        _partial_memcheck_has_application_pass(sanitizer, str(case.get("gpu_uuid", "")))
    )
    error_count = sanitizer.get("error_count")
    if (type(error_count) is not int or error_count != 0 or
            not (sanitizer_passed or partial_memcheck_passed)):
        raise EvidenceError(f"{case['id']}: sanitizer reported errors or incomplete execution")

    profile_record = runs["profile_candidate"].get("profile")
    if not isinstance(profile_record, dict):
        raise EvidenceError(f"{case['id']}: selected NCU profile is missing")
    report_path = checked_file(base, profile_record.get("report"), f"{case['id']} NCU report")
    analysis_path = checked_file(base, profile_record.get("analysis"), f"{case['id']} NCU analysis")
    shim_log_path = checked_file(base, profile_record.get("shim_log"), f"{case['id']} NCU shim log")
    analysis = read_json(analysis_path)
    selected_kernel = profile_record.get("selected_kernel")
    if (not isinstance(selected_kernel, str) or not selected_kernel or
            analysis.get("selected_kernel") != selected_kernel or analysis.get("readable") is not True or
            not report_path.stat().st_size):
        raise EvidenceError(f"{case['id']}: NCU report is empty or does not identify a readable selected kernel")
    ncu_identity = summaries["profile_candidate"].get("provenance", {}).get("tools", {}).get("ncu")
    if (not isinstance(ncu_identity, dict) or not ncu_identity.get("path") or
            not ncu_identity.get("version")):
        raise EvidenceError(f"{case['id']}: NCU tool identity is missing from controller provenance")
    try:
        shim_records = [json.loads(line) for line in shim_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{case['id']}: malformed NCU shim invocation log") from exc
    collection = [item for item in shim_records if isinstance(item, dict) and
                  item.get("policy") == "collection-clock-control-none-source-import-yes"]
    shim_path = absolute_repo_file(root, "benchmark_runs/volta-environment/ncu-collection-shim.py",
                                   f"{case['id']} checked-in NCU shim")
    shim_sha = digest(shim_path)
    ncu_path = Path(str(ncu_identity["path"])).resolve()
    profile_inputs = _input_bindings(summaries["profile_candidate"])
    if not collection or not all(
        isinstance(item.get("argv_executed"), list) and
        _argument_values(item["argv_executed"], "--clock-control") == ["none"] and
        _argument_values(item["argv_executed"], "--import-source") == ["yes"] and
        isinstance(item.get("shim_path"), str) and Path(item["shim_path"]).resolve() == shim_path and
        item.get("shim_sha256") == shim_sha and
        profile_inputs.get(str(shim_path)) == shim_sha and
        isinstance(item.get("actual_ncu_path"), str) and Path(item["actual_ncu_path"]).resolve() == ncu_path and
        ncu_path.is_file() and item.get("actual_ncu_sha256") == digest(ncu_path) and
        profile_inputs.get(str(ncu_path)) == item.get("actual_ncu_sha256")
        for item in collection
    ):
        raise EvidenceError(f"{case['id']}: NCU collection did not record the no-clock/source-import policy")


def _argument_values(argv: list[Any], flag: str) -> list[str]:
    values = []
    for index, value in enumerate(argv):
        if value == flag:
            if index + 1 < len(argv) and isinstance(argv[index + 1], str):
                values.append(argv[index + 1])
            else:
                values.append("")
        if isinstance(value, str) and value.startswith(flag + "="):
            values.append(value.split("=", 1)[1])
    return values


def validate(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise EvidenceError(f"manifest format must be {FORMAT}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvidenceError("manifest contains no cases")
    root = repo_root(path.parent)
    evidence_ids: set[str] = set()
    errors: list[str] = []
    checked = 0
    for case in cases:
        try:
            _check_case(path.parent.resolve(), root, case, evidence_ids)
            checked += 1
        except EvidenceError as exc:
            errors.append(str(exc))
    return {"format": "KERNELWIKI-VOLTA-EVIDENCE-CHECK/1", "valid": not errors,
            "checked_cases": checked, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--manifest", type=Path, help="validate one explicit evidence manifest")
    selection.add_argument("--case", choices=sorted(CASE_PATHS), help="validate a named canonical case manifest")
    selection.add_argument("--all", action="store_true", help="validate all canonical case manifests")
    args = parser.parse_args(argv)

    if args.manifest is not None:
        paths = {"manifest": args.manifest.resolve()}
    else:
        root = Path(__file__).resolve().parents[1]
        selected = CASE_PATHS if args.all else {args.case: CASE_PATHS[args.case]}
        paths = {name: root / relative for name, relative in selected.items()}

    results = {}
    for name, path in paths.items():
        try:
            results[name] = validate(path)
        except EvidenceError as exc:
            results[name] = {"format": "KERNELWIKI-VOLTA-EVIDENCE-CHECK/1", "valid": False,
                             "checked_cases": 0, "errors": [str(exc)]}
    if args.all:
        result = {"format": "KERNELWIKI-VOLTA-EVIDENCE-CHECK/1",
                  "valid": all(item["valid"] for item in results.values()),
                  "cases": results,
                  "checked_cases": sum(item["checked_cases"] for item in results.values()),
                  "errors": [f"{name}: {error}" for name, item in results.items()
                             for error in item["errors"]]}
    else:
        result = next(iter(results.values()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
