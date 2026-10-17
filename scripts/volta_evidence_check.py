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
GEMM_OPERATOR = "C[m,n] = sum_k A16[m,k] * B16[k,n]"
GEMM_CHECK_FORMAT = "KERNELWIKI-GEMM-CHECK/1"
GEMM_PRIMARY_ROLES = {
    "correctness_baseline": ("baseline", "cublas", "correctness", None),
    "correctness_candidate": ("candidate", "ptx", "correctness", "2"),
    "sanitizer_candidate": ("candidate", "ptx", "correctness", "2"),
    "baseline_timing": ("baseline", "cublas", "time", None),
    "candidate_timing": ("candidate", "ptx", "time", "2"),
    "profile_candidate": ("candidate", "ptx", "single", "2"),
}
ROW_CONTRACT_FORMAT = "KERNELWIKI-ROW-CONTRACT/1"
ROW_CHECK_FORMAT = "KERNELWIKI-ROW-CHECK/1"
ROW_TIMING_FORMAT = "KERNELWIKI-ROW-TIMING/1"
ROW_PROFILE_CASE_ID = "softmax_wide_streaming"
ROW_PRIMARY_ROLES = {
    "correctness_baseline": ("baseline", "correctness", None, None, None, None),
    "correctness_candidate": ("candidate", "correctness", None, None, None, None),
    "sanitizer_candidate": ("candidate", "correctness", None, None, None, None),
    "baseline_timing": ("baseline", "time", "timing", 5, 21, 2),
    "candidate_timing": ("candidate", "time", "timing", 5, 21, 2),
    "profile_candidate": ("candidate", "time", "profile", 0, 1, 1),
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


def _is_gemm_contract(contract: dict[str, Any]) -> bool:
    numerical_contract = contract.get("numerical_contract")
    return (isinstance(numerical_contract, dict) and
            numerical_contract.get("operator") == GEMM_OPERATOR)


def _is_rows_contract(contract: dict[str, Any]) -> bool:
    return contract.get("format") == ROW_CONTRACT_FORMAT


def _row_contract_cases(contract: dict[str, Any]) -> list[dict[str, Any]]:
    if (contract.get("case_id") != "rows" or
            contract.get("suite_id") != "rows_correctness_suite" or
            contract.get("operators") != ["softmax", "rmsnorm"]):
        raise EvidenceError("rows contract does not identify the fixed softmax/RMSNorm suite")
    cases = contract.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvidenceError("rows contract per-case declarations are missing")
    ids: list[str] = []
    allowed_paths = {
        "softmax": ({"softmax_warp"}, {"softmax_cta"}),
        "rmsnorm": ({"rmsnorm_cta_scalar"}, {"vec4", "scalar_fallback"}),
    }
    for item in cases:
        if not isinstance(item, dict):
            raise EvidenceError("rows contract contains a malformed case declaration")
        case_id, operation = item.get("case_id"), item.get("operation")
        rows, cols, expected = item.get("rows"), item.get("cols"), item.get("expected_outputs")
        if (not isinstance(case_id, str) or not case_id or case_id in ids or
                not isinstance(operation, str) or operation not in allowed_paths or
                type(rows) is not int or rows < 0 or
                type(cols) is not int or cols <= 0 or type(expected) is not int or
                expected != rows * cols):
            raise EvidenceError("rows contract contains invalid case identity or dimensions")
        baseline_path, candidate_path = item.get("baseline_path"), item.get("candidate_path")
        if rows == 0:
            if baseline_path != "no_launch" or candidate_path != "no_launch":
                raise EvidenceError(f"rows contract {case_id}: zero-row route must be no_launch")
        else:
            baseline_routes, candidate_routes = allowed_paths[operation]
            if (not isinstance(baseline_path, str) or not isinstance(candidate_path, str) or
                    baseline_path not in baseline_routes or candidate_path not in candidate_routes):
                raise EvidenceError(f"rows contract {case_id}: implementation route is invalid")
        if item.get("check_identity") != f"logical-row-major:{case_id}":
            raise EvidenceError(f"rows contract {case_id}: output check identity is invalid")
        ids.append(case_id)
    if contract.get("case_ids") != ids or contract.get("correctness_case_ids") != ids:
        raise EvidenceError("rows contract case ID lists do not match its ordered declarations")
    return cases


def _validate_rows_oracle_contract(contract: dict[str, Any], oracle: dict[str, Any]) -> list[dict[str, Any]]:
    cases = _row_contract_cases(contract)
    checks = oracle.get("checks")
    if not isinstance(checks, list) or len(checks) != len(cases):
        raise EvidenceError("rows CPU oracle case checks do not match the contract suite")
    for spec, check in zip(cases, checks):
        if (not isinstance(check, dict) or
                check.get("case_id") != spec["case_id"] or
                check.get("operation") != spec["operation"] or
                type(check.get("rows")) is not int or
                check.get("rows") != spec["rows"] or
                type(check.get("cols")) is not int or
                check.get("cols") != spec["cols"] or
                type(check.get("reference_values")) is not int or
                check.get("reference_values") != spec["expected_outputs"] or
                check.get("passed") is not True):
            raise EvidenceError(f"rows CPU oracle does not match contract case {spec['case_id']}")
    return cases


def _rows_contract_artifact(base: Path, contract_path: Path, record: Any,
                            label: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise EvidenceError(f"{label}: rows contract artifact path/hash is missing")
    raw_path, expected = record.get("path"), record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected, str):
        raise EvidenceError(f"{label}: rows contract artifact path/hash is malformed")
    path = (contract_path.parent / raw_path).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise EvidenceError(f"{label}: rows contract artifact is missing or outside the evidence directory")
    if digest(path) != expected:
        raise EvidenceError(f"{label}: rows contract artifact SHA-256 mismatch")
    return path, expected


def _validate_rows_contract_artifacts(base: Path, contract_path: Path,
                                      contract: dict[str, Any], input_path: Path,
                                      input_sha: str, oracle_path: Path,
                                      oracle_sha: str, oracle: dict[str, Any]) -> tuple[Path, str]:
    represented_path, represented_sha = _rows_contract_artifact(
        base, contract_path, contract.get("represented_input"), "rows represented input")
    if represented_path.resolve() != input_path.resolve() or represented_sha != input_sha:
        raise EvidenceError("rows contract does not bind the manifest represented input")
    reference_path, reference_sha = _rows_contract_artifact(
        base, contract_path, contract.get("reference_output"), "rows reference output")
    oracle_contract_path, oracle_contract_sha = _rows_contract_artifact(
        base, contract_path, contract.get("cpu_oracle"), "rows CPU oracle")
    if oracle_contract_path.resolve() != oracle_path.resolve() or oracle_contract_sha != oracle_sha:
        raise EvidenceError("rows contract does not bind the manifest CPU oracle")
    if (oracle.get("represented_input_sha256") != input_sha or
            oracle.get("reference_sha256") != reference_sha):
        raise EvidenceError("rows CPU oracle hashes do not match represented input/reference output")
    return reference_path, reference_sha


def _rows_native_pass_matches_contract(result: dict[str, Any], contract: dict[str, Any],
                                       gpu_uuid: str, implementation: str) -> bool:
    try:
        cases = _row_contract_cases(contract)
    except EvidenceError:
        return False
    expected_uuid = gpu_uuid.removeprefix("GPU-").replace("-", "").lower()
    if (result.get("format") != ROW_CHECK_FORMAT or result.get("status") != "passed" or
            result.get("operator") != "row_suite" or
            result.get("implementation") != implementation or
            result.get("case_scope") != "full_suite" or
            result.get("compute_capability") != "7.0" or
            result.get("gpu_uuid") != expected_uuid or
            type(result.get("case_count")) is not int or
            result.get("case_count") != len(cases)):
        return False
    expected_total = sum(item["expected_outputs"] for item in cases)
    if (type(result.get("expected_outputs")) is not int or
            result.get("expected_outputs") != expected_total or
            type(result.get("checked_outputs")) is not int or
            result.get("checked_outputs") != expected_total or
            any(type(result.get(field)) is not int or result[field] != 0
                for field in ("failed_outputs", "nonfinite_outputs", "padding_writes"))):
        return False
    observed = result.get("cases")
    if not isinstance(observed, list) or len(observed) != len(cases):
        return False
    common = contract.get("numerical_contract", {}).get("common", {})
    atol, rtol = common.get("tolerance_atol"), common.get("tolerance_rtol")
    if not (_finite_nonnegative_number(atol) and _finite_nonnegative_number(rtol)):
        return False
    route_key = "baseline_path" if implementation == "baseline" else "candidate_path"
    for spec, item in zip(cases, observed):
        if not isinstance(item, dict):
            return False
        if any(type(item.get(field)) is not int for field in ("rows", "cols", "expected_outputs")):
            return False
        if any(item.get(field) != spec[field] for field in
               ("case_id", "operation", "rows", "cols", "expected_outputs", "check_identity")):
            return False
        if (item.get("implementation_path") != spec[route_key] or
                item.get("passed") is not True or
                any(type(item.get(field)) is not int or item[field] != 0
                    for field in ("failed_outputs", "nonfinite_outputs", "padding_writes", "failed_checks")) or
                type(item.get("checked_outputs")) is not int or
                item.get("checked_outputs") != spec["expected_outputs"] or
                type(item.get("padding_checked")) is not int or item.get("padding_checked") < 0):
            return False
        metrics = ("max_abs_error", "max_admission_tolerance", "max_tolerance_ratio")
        if (not all(_finite_nonnegative_number(item.get(field)) for field in metrics) or
                float(item["max_abs_error"]) > float(item["max_admission_tolerance"]) or
                float(item["max_tolerance_ratio"]) > 1.0 or
                not _finite_nonnegative_number(item.get("atol")) or
                not _finite_nonnegative_number(item.get("rtol")) or
                float(item["atol"]) != float(atol) or float(item["rtol"]) != float(rtol)):
            return False
    return True


def _row_option(argv: list[Any], role: str, flag: str, expected: str | None) -> None:
    occurrences = [i for i, item in enumerate(argv)
                   if isinstance(item, str) and (item == flag or item.startswith(flag + "="))]
    expected_count = 0 if expected is None else 1
    if len(occurrences) != expected_count:
        raise EvidenceError(f"{role}: row controller argv must contain {expected_count} {flag} selector(s)")
    if expected is None:
        return
    index = occurrences[0]
    if (argv[index] != flag or index + 1 >= len(argv) or
            not isinstance(argv[index + 1], str) or argv[index + 1].startswith("--") or
            argv[index + 1] != expected):
        raise EvidenceError(f"{role}: row controller argv {flag} selector does not match the frozen role")


def _validate_rows_primary_argv(summaries: dict[str, dict[str, Any]], root: Path,
                                contract: dict[str, Any], input_path: Path,
                                input_sha: str, reference_path: Path,
                                reference_sha: str,
                                implementations: dict[str, dict[str, Any]]) -> None:
    cases = _row_contract_cases(contract)
    timing_cases = contract.get("selected_timing_case_ids")
    if (not isinstance(timing_cases, list) or not timing_cases or
            any(not isinstance(item, str) for item in timing_cases) or
            timing_cases != [item["case_id"] for item in cases if item["case_id"] in timing_cases]):
        raise EvidenceError("rows contract selected timing case IDs are invalid")
    role_selectors = {}
    for role, (implementation, mode, selection, warmups, repetitions, rotations) in ROW_PRIMARY_ROLES.items():
        if selection == "timing":
            case_selector = ",".join(timing_cases)
        elif selection == "profile":
            case_selector = ROW_PROFILE_CASE_ID
            if case_selector not in timing_cases:
                raise EvidenceError("rows profile case is not in selected timing cases")
        else:
            case_selector = None
        role_selectors[role] = (implementation, mode, case_selector, warmups, repetitions, rotations)

    for role, (implementation, mode, case_selector, warmups, repetitions, rotations) in role_selectors.items():
        summary = summaries.get(role)
        provenance = summary.get("provenance") if isinstance(summary, dict) else None
        argv = provenance.get("argv") if isinstance(provenance, dict) else None
        if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
            raise EvidenceError(f"{role}: controller provenance argv is missing for rows")
        source = provenance.get("source")
        project_root = source.get("project_root") if isinstance(source, dict) else None
        if not isinstance(project_root, str) or not project_root:
            raise EvidenceError(f"{role}: controller source project_root is missing for rows argv")
        capture_root = Path(project_root).resolve()
        executable = Path(argv[0])
        if not executable.is_absolute():
            executable = capture_root / executable
        binary = provenance.get("binary")
        binary_raw = binary.get("path") if isinstance(binary, dict) else None
        if not isinstance(binary_raw, str) or not binary_raw:
            raise EvidenceError(f"{role}: controller binary path is missing for rows argv")
        provenance_binary = Path(binary_raw)
        if not provenance_binary.is_absolute():
            provenance_binary = capture_root / provenance_binary
        implementation_record = implementations.get(implementation)
        expected_executable = implementation_record.get("executed_path") if isinstance(implementation_record, dict) else None
        if (not isinstance(expected_executable, Path) or
                executable.resolve() != expected_executable.resolve() or
                provenance_binary.resolve() != expected_executable.resolve()):
            raise EvidenceError(f"{role}: controller argv[0] differs from the declared rows executable")

        expected_input = str(input_path.resolve().relative_to(root.resolve()))
        expected_reference = str(reference_path.resolve().relative_to(root.resolve()))
        _row_option(argv, role, "--input", expected_input)
        _row_option(argv, role, "--reference", expected_reference)
        _row_option(argv, role, "--impl", implementation)
        _row_option(argv, role, "--mode", mode)
        _row_option(argv, role, "--case", case_selector)
        _row_option(argv, role, "--warmups", str(warmups) if warmups is not None else None)
        _row_option(argv, role, "--repetitions", str(repetitions) if repetitions is not None else None)
        _row_option(argv, role, "--rotations", str(rotations) if rotations is not None else None)
        expected_argv = ["--input", expected_input, "--reference", expected_reference,
                         "--impl", implementation, "--mode", mode]
        if case_selector is not None:
            expected_argv.extend(("--case", case_selector))
        if warmups is not None:
            expected_argv.extend(("--warmups", str(warmups), "--repetitions", str(repetitions),
                                  "--rotations", str(rotations)))
        if argv[1:] != expected_argv:
            raise EvidenceError(f"{role}: controller argv differs from the frozen rows role")
        if _input_bindings(summary).get(str(reference_path.resolve())) != reference_sha:
            raise EvidenceError(f"{role}: controller inputs do not bind the rows reference output")


def _read_rows_stdout(base: Path, run: Any, label: str) -> dict[str, Any]:
    if not isinstance(run, dict):
        raise EvidenceError(f"{label}: row runner stdout receipt is missing")
    path = checked_file(base, run.get("stdout"), f"{label} row runner stdout")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{label}: row runner stdout is not one JSON object") from exc
    if not isinstance(result, dict):
        raise EvidenceError(f"{label}: row runner stdout is not a JSON object")
    return result


def _validate_rows_timing_output(result: dict[str, Any], contract: dict[str, Any],
                                 gpu_uuid: str, role: str) -> None:
    selectors = ROW_PRIMARY_ROLES[role]
    implementation, _mode, selection, warmups, repetitions, rotations = selectors
    expected_ids = (contract.get("selected_timing_case_ids") if selection == "timing"
                    else [ROW_PROFILE_CASE_ID])
    specs = {item["case_id"]: item for item in _row_contract_cases(contract)}
    if (not isinstance(expected_ids, list) or any(item not in specs for item in expected_ids) or
            result.get("format") != ROW_TIMING_FORMAT or result.get("status") != "completed" or
            result.get("implementation") != implementation or
            result.get("gpu_uuid") != gpu_uuid.removeprefix("GPU-").replace("-", "").lower() or
            result.get("compute_capability") != "7.0" or
            type(result.get("warmups")) is not int or result.get("warmups") != warmups or
            type(result.get("repetitions")) is not int or result.get("repetitions") != repetitions or
            type(result.get("rotation_count")) is not int or result.get("rotation_count") != rotations):
        raise EvidenceError(f"{role}: native rows timing header differs from the frozen role")
    observed = result.get("cases")
    if not isinstance(observed, list) or len(observed) != len(expected_ids):
        raise EvidenceError(f"{role}: native rows timing cases do not match the selected role")
    path_key = "baseline_path" if implementation == "baseline" else "candidate_path"
    for case_id, item in zip(expected_ids, observed):
        spec = specs[case_id]
        samples = item.get("samples_us") if isinstance(item, dict) else None
        if (not isinstance(item, dict) or item.get("case_id") != case_id or
                item.get("operation") != spec["operation"] or
                type(item.get("rows")) is not int or type(item.get("cols")) is not int or
                item.get("rows") != spec["rows"] or item.get("cols") != spec["cols"] or
                item.get("implementation_path") != spec[path_key] or
                not isinstance(samples, list) or len(samples) != repetitions or
                any(not _finite_positive_number(value) for value in samples)):
            raise EvidenceError(f"{role}: native rows timing output is invalid for {case_id}")


def _finite_positive_number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(float(value)) and float(value) > 0)


def _finite_nonnegative_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def _gemm_native_pass_matches_contract(result: dict[str, Any],
                                       contract: dict[str, Any]) -> bool:
    if (result.get("format") != GEMM_CHECK_FORMAT or
            result.get("operator") != "gemm" or
            result.get("implementation") != "ptx" or
            type(result.get("shared_stages")) is not int or
            result.get("shared_stages") != 2 or
            result.get("case_id") != contract.get("case_id") or
            result.get("passed") is not True):
        return False
    shape = contract.get("shape")
    if not isinstance(shape, dict):
        return False
    m, n = shape.get("m"), shape.get("n")
    if (type(m) is not int or m <= 0 or type(n) is not int or n <= 0 or
            type(result.get("checked_outputs")) is not int or
            result["checked_outputs"] != m * n):
        return False
    if any(type(result.get(field)) is not int or result[field] != 0
           for field in ("failed_outputs", "nonfinite_outputs")):
        return False
    metrics = ("max_abs_error", "max_admission_tolerance", "max_tolerance_ratio")
    if not all(_finite_nonnegative_number(result.get(field)) for field in metrics):
        return False
    return (float(result["max_abs_error"]) <= float(result["max_admission_tolerance"]) and
            float(result["max_tolerance_ratio"]) <= 1.0)


def _gemm_option(argv: list[Any], role: str, flag: str,
                 expected: str | None) -> str | None:
    occurrences = [
        index for index, item in enumerate(argv)
        if isinstance(item, str) and
        (item == flag or item.startswith(flag + "="))
    ]
    expected_count = 0 if expected is None else 1
    if len(occurrences) != expected_count:
        raise EvidenceError(
            f"{role}: controller argv must contain {expected_count} {flag} selector(s)"
        )
    if not occurrences:
        return None
    index = occurrences[0]
    if (argv[index] != flag or index + 1 >= len(argv) or
            not isinstance(argv[index + 1], str) or argv[index + 1].startswith("--")):
        raise EvidenceError(f"{role}: controller argv has malformed {flag} selector")
    value = argv[index + 1]
    if value != expected:
        raise EvidenceError(f"{role}: controller argv {flag} selector does not match GEMM contract")
    return value


def _validate_gemm_input_option(argv: list[Any], role: str, root: Path,
                                expected_path: Path) -> None:
    occurrences = [
        index for index, item in enumerate(argv)
        if isinstance(item, str) and
        (item == "--input" or item.startswith("--input="))
    ]
    if len(occurrences) != 1:
        raise EvidenceError(f"{role}: controller argv must contain exactly one --input selector")
    index = occurrences[0]
    if (argv[index] != "--input" or index + 1 >= len(argv) or
            not isinstance(argv[index + 1], str) or argv[index + 1].startswith("--")):
        raise EvidenceError(f"{role}: controller argv has malformed --input selector")
    input_arg = Path(argv[index + 1])
    if not input_arg.is_absolute():
        input_arg = root / input_arg
    if input_arg.resolve() != expected_path.resolve():
        raise EvidenceError(f"{role}: controller argv --input does not resolve to represented GEMM input")


def _validate_gemm_primary_argv(summaries: dict[str, dict[str, Any]], root: Path,
                               contract: dict[str, Any], input_path: Path,
                               implementations: dict[str, dict[str, Any]]) -> None:
    """Bind the six primary GEMM run roles to their declared runner selectors."""
    case_id = contract.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise EvidenceError("GEMM contract case_id is missing")
    for role, (implementation_name, implementation, mode, stages) in GEMM_PRIMARY_ROLES.items():
        summary = summaries.get(role)
        provenance = summary.get("provenance") if isinstance(summary, dict) else None
        argv = provenance.get("argv") if isinstance(provenance, dict) else None
        if not isinstance(argv, list):
            raise EvidenceError(f"{role}: controller provenance argv is missing for GEMM")
        source = provenance.get("source")
        project_root = source.get("project_root") if isinstance(source, dict) else None
        if not isinstance(project_root, str) or not project_root:
            raise EvidenceError(f"{role}: controller source project_root is missing for GEMM argv")
        if not argv or not isinstance(argv[0], str) or not argv[0]:
            raise EvidenceError(f"{role}: controller argv executable is missing for GEMM")
        capture_root = Path(project_root).resolve()
        argv_executable = Path(argv[0])
        if not argv_executable.is_absolute():
            argv_executable = capture_root / argv_executable
        binary = provenance.get("binary")
        binary_raw = binary.get("path") if isinstance(binary, dict) else None
        if not isinstance(binary_raw, str) or not binary_raw:
            raise EvidenceError(f"{role}: controller binary path is missing for GEMM argv")
        provenance_binary = Path(binary_raw)
        if not provenance_binary.is_absolute():
            provenance_binary = capture_root / provenance_binary
        implementation_record = implementations.get(implementation_name)
        expected_executable = (
            implementation_record.get("executed_path")
            if isinstance(implementation_record, dict) else None
        )
        if (not isinstance(expected_executable, Path) or
                argv_executable.resolve() != expected_executable.resolve() or
                provenance_binary.resolve() != expected_executable.resolve()):
            raise EvidenceError(
                f"{role}: controller argv[0] differs from the declared GEMM executable"
            )
        _gemm_option(argv, role, "--case", case_id)
        _validate_gemm_input_option(argv, role, root, input_path)
        _gemm_option(argv, role, "--impl", implementation)
        _gemm_option(argv, role, "--mode", mode)
        _gemm_option(argv, role, "--stages", stages)


def _partial_memcheck_has_application_pass(summary: dict[str, Any], gpu_uuid: str,
                                            contract: dict[str, Any] | None = None) -> bool:
    """Accept only the known partial memcheck classification with a raw target PASS."""
    if (summary.get("tool") != "compute-sanitizer" or
            summary.get("status") != "partial" or
            summary.get("tool_detail") != "memcheck" or
            type(summary.get("exit_status")) is not int or
            summary.get("exit_status") != 0 or summary.get("conclusive") is not False):
        return False
    if type(summary.get("error_count")) is not int or summary.get("error_count") != 0:
        return False
    expected_uuid = gpu_uuid.removeprefix("GPU-").replace("-", "").lower()
    lines = summary.get("stdout_tail")
    if not isinstance(lines, list):
        return False
    gemm_contract = isinstance(contract, dict) and _is_gemm_contract(contract)
    rows_contract = isinstance(contract, dict) and _is_rows_contract(contract)
    for line in lines:
        if not isinstance(line, str):
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if gemm_contract:
            if isinstance(result, dict) and _gemm_native_pass_matches_contract(result, contract):
                return True
            continue
        if rows_contract:
            if (isinstance(result, dict) and
                    _rows_native_pass_matches_contract(result, contract, gpu_uuid, "candidate")):
                return True
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
    rows_reference_path: Path | None = None
    rows_reference_sha: str | None = None
    if _is_rows_contract(contract_data):
        _validate_rows_oracle_contract(contract_data, oracle)
        rows_reference_path, rows_reference_sha = _validate_rows_contract_artifacts(
            base, contract, contract_data, input_path, input_sha, oracle_path,
            oracle_sha, oracle,
        )

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
    if _is_gemm_contract(contract_data):
        _validate_gemm_primary_argv(summaries, root, contract_data, input_path, bound_impls)
    elif _is_rows_contract(contract_data):
        if rows_reference_path is None or rows_reference_sha is None:
            raise EvidenceError(f"{case['id']}: rows reference binding is missing")
        _validate_rows_primary_argv(
            summaries, root, contract_data, input_path, input_sha,
            rows_reference_path, rows_reference_sha, bound_impls,
        )
        for role, implementation in (("correctness_baseline", "baseline"),
                                     ("correctness_candidate", "candidate")):
            native = _read_rows_stdout(base, runs.get(role), f"{case['id']} {role}")
            if not _rows_native_pass_matches_contract(
                    native, contract_data, str(case.get("gpu_uuid", "")), implementation):
                raise EvidenceError(f"{case['id']} {role}: native rows correctness receipt does not pass the frozen contract")
        for role in ("baseline_timing", "candidate_timing", "profile_candidate"):
            native = _read_rows_stdout(base, runs.get(role), f"{case['id']} {role}")
            _validate_rows_timing_output(native, contract_data, str(case.get("gpu_uuid", "")), role)
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
        _partial_memcheck_has_application_pass(
            sanitizer, str(case.get("gpu_uuid", "")), contract_data
        )
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
