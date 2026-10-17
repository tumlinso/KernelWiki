#!/usr/bin/env python3
"""Fail-closed checks for source-bound, controller-backed Volta comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from pathlib import PurePosixPath
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
ATTENTION_CONTRACT_FORMAT = "KERNELWIKI-VOLTA-ATTENTION-CONTRACT/1"
ATTENTION_RUN_FORMAT = "KERNELWIKI-ATTENTION-RUN/1"
ROW_PROFILE_CASE_ID = "softmax_wide_streaming"
ROW_CAPTURED_SOURCE_PATHS = frozenset({
    "artifacts/kernels/volta-rows/full/PROVENANCE.yaml",
    "artifacts/kernels/volta-rows/full/README.md",
    "artifacts/kernels/volta-rows/full/row_kernels.cu",
    "artifacts/kernels/volta-rows/full/row_runner.cu",
})
ROW_PRIMARY_ROLES = {
    "correctness_baseline": ("baseline", "correctness", None, None, None, None),
    "correctness_candidate": ("candidate", "correctness", None, None, None, None),
    "sanitizer_candidate": ("candidate", "correctness", None, None, None, None),
    "baseline_timing": ("baseline", "time", "timing", 5, 21, 2),
    "candidate_timing": ("candidate", "time", "timing", 5, 21, 2),
    "profile_candidate": ("candidate", "time", "profile", 0, 1, 1),
}
ATTENTION_REQUIRED_ROLES = {
    "correctness_baseline": ("baseline", "baseline", "foreground-baseline", "correctness", "all"),
    "correctness_candidate": ("candidate", "online", "foreground-baseline", "correctness", "all"),
    "sanitizer_candidate": ("candidate", "online", "foreground-compute-sanitizer", "correctness", "all"),
    "baseline_prefill_timing": ("baseline", "baseline", "foreground-baseline", "time", "prefill64-long-causal"),
    "candidate_prefill_timing": ("candidate", "online", "foreground-baseline", "time", "prefill64-long-causal"),
    "baseline_decode_timing": ("baseline", "baseline", "foreground-baseline", "time", "decode70-causal"),
    "candidate_decode_timing": ("candidate", "online", "foreground-baseline", "time", "decode70-causal"),
    "profile_candidate_decode": ("candidate", "online", "foreground-ncu", "correctness", "decode70-causal"),
}
ATTENTION_OPTIONAL_ROLES = {
    "correctness_split_decode": ("split", "split", "foreground-baseline", "correctness", "decode70-causal"),
    "sanitizer_split_decode": ("split", "split", "foreground-compute-sanitizer", "correctness", "decode70-causal"),
    "split_decode_timing": ("split", "split", "foreground-baseline", "time", "decode70-causal"),
}
ATTENTION_ARCHIVE_ROLE_ALIASES = {
    "correctness_split_decode": "correctness_split16",
    "sanitizer_split_decode": "sanitizer_split16",
    "split_decode_timing": "split16_decode_timing",
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


def _repo_relative_identity(root: Path, raw_path: Any, label: str) -> Path:
    """Resolve a captured repository path without requiring the old file to remain present."""
    if (not isinstance(raw_path, str) or not raw_path or "\\" in raw_path or
            PurePosixPath(raw_path).is_absolute() or
            any(part in {"", ".", ".."} for part in raw_path.split("/"))):
        raise EvidenceError(f"{label}: expected a safe repository-relative path")
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise EvidenceError(f"{label}: repository-relative path escapes the repository")
    return path


def _captured_input_archives(root: Path, case: dict[str, Any], required: bool
                             ) -> dict[str, dict[str, Any]]:
    raw_archives = case.get("captured_input_archives")
    if raw_archives is None:
        if required:
            raise EvidenceError(f"{case.get('id', 'case')}: captured input archive bindings are missing")
        return {}
    if not isinstance(raw_archives, list) or not raw_archives:
        raise EvidenceError(f"{case.get('id', 'case')}: captured input archive bindings are malformed")

    archives: dict[str, dict[str, Any]] = {}
    archive_paths: set[str] = set()
    for index, item in enumerate(raw_archives):
        label = f"{case.get('id', 'case')} captured input archive {index}"
        if not isinstance(item, dict):
            raise EvidenceError(f"{label}: expected original_path, archive_path, and sha256")
        original_path = _repo_relative_identity(root, item.get("original_path"), f"{label} original path")
        archive_path = item.get("archive_path")
        archive_resolved = _repo_relative_identity(root, archive_path, f"{label} archive path")
        raw_sha = item.get("sha256")
        if not isinstance(raw_sha, str) or len(raw_sha) != 64:
            raise EvidenceError(f"{label}: SHA-256 is missing or malformed")
        archive_file = absolute_repo_file(root, archive_path, f"{label} archive")
        if digest(archive_file) != raw_sha:
            raise EvidenceError(f"{label}: archive SHA-256 mismatch")
        original_key = str(original_path)
        archive_key = str(archive_resolved)
        if original_key in archives or archive_key in archive_paths:
            raise EvidenceError(f"{label}: duplicate original or archive path")
        archive_paths.add(archive_key)
        archives[original_key] = {
            "original_path": original_path,
            "archive_path": archive_file,
            "archive_path_text": archive_path,
            "sha256": raw_sha,
        }
    return archives


def _exact_input_bindings(summary: dict[str, Any], expected: dict[str, str],
                          allow_duplicate_paths: set[str] | None = None) -> bool:
    """Require every archived capture alias with its declared digest.

    A small set of attention bundles are intentionally listed both as a frozen
    case input and as a Project Control spec input. Their duplicate provenance
    rows are accepted only when every occurrence carries the same digest.
    """
    provenance = summary.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, dict) else None
    if not isinstance(inputs, list):
        return False
    found: dict[str, list[str]] = {path: [] for path in expected}
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        resolved = str(Path(item["path"]).resolve())
        if resolved in found:
            found[resolved].append(item.get("sha256"))
    allowed = allow_duplicate_paths or set()
    return all(
        bool(found[path]) and all(value == sha for value in found[path])
        if path in allowed else found[path] == [sha]
        for path, sha in expected.items()
    )


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
                    evidence_ids: set[str], captured_inputs: dict[str, str] | None = None,
                    additional_inputs: dict[str, str] | None = None,
                    allow_duplicate_captured_inputs: set[str] | None = None
                    ) -> dict[str, Any]:
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
        str(implementation["source_input_path"].resolve()): implementation["source_sha"],
    }
    if captured_inputs:
        expected_inputs.update(captured_inputs)
    if additional_inputs:
        expected_inputs.update(additional_inputs)
    actual_inputs = _input_bindings(summary)
    if any(actual_inputs.get(path) != value for path, value in expected_inputs.items()):
        raise EvidenceError(f"{label}: controller inputs do not bind the contract, represented data, and source")
    if captured_inputs and not _exact_input_bindings(
            summary, captured_inputs, allow_duplicate_captured_inputs):
        raise EvidenceError(f"{label}: controller inputs do not exactly bind archived source aliases")
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


def _is_attention_contract(contract: dict[str, Any]) -> bool:
    return contract.get("format") == ATTENTION_CONTRACT_FORMAT


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


def _attention_case_dimensions(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    signature = contract.get("case_signature")
    records = signature.get("cases") if isinstance(signature, dict) else None
    if not isinstance(records, list) or not records:
        raise EvidenceError("attention contract has no frozen case signature")
    result: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise EvidenceError("attention contract contains a malformed case signature")
        case_id = item["id"]
        q_shape, kv_shape = item.get("q_shape"), item.get("kv_shape")
        if (case_id in result or not isinstance(q_shape, list) or len(q_shape) != 4 or
                not isinstance(kv_shape, list) or len(kv_shape) != 4 or
                any(type(dimension) is not int or dimension <= 0
                    for dimension in (*q_shape, *kv_shape))):
            raise EvidenceError(f"attention case signature is malformed: {case_id}")
        result[case_id] = {
            "outputs": math.prod(q_shape),
            "rows": q_shape[0] * q_shape[1] * q_shape[2],
            "shape": q_shape,
            "kv_shape": kv_shape,
            "signature": item,
        }
    if set(result) != {
        "prefill64-causal", "prefill64-medium-causal", "prefill64-long-causal",
        "prefill128-noncausal", "decode70-causal", "selected64-causal",
        "score-stress64-noncausal",
    }:
        raise EvidenceError("attention contract case set differs from the frozen seven-case suite")
    return result


def _attention_nonnegative(value: Any) -> bool:
    return _finite_nonnegative_number(value)


def _attention_native_checks(result: dict[str, Any], contract: dict[str, Any],
                             oracle_data: dict[str, dict[str, Any]], gpu_uuid: str,
                             implementation: str, mode: str, selected_case: str,
                             splits: int, label: str) -> None:
    raw_uuid = result.get("gpu_uuid")
    if (result.get("format") != ATTENTION_RUN_FORMAT or result.get("status") != "passed" or
            str(result.get("compute_capability")) != "7.0" or
            not isinstance(raw_uuid, str) or
            raw_uuid.lower() != gpu_uuid.removeprefix("GPU-").replace("-", "").lower() or
            result.get("case_id") != "attention" or result.get("case") != selected_case or
            result.get("mode") != mode or result.get("implementation") != implementation or
            type(result.get("splits")) is not int or result.get("splits") != splits or
            result.get("failure") is not None):
        raise EvidenceError(f"{label}: native attention result identity or status differs from its role")

    specs = _attention_case_dimensions(contract)
    expected_ids = list(specs) if selected_case == "all" else [selected_case]
    if any(case_id not in specs for case_id in expected_ids):
        raise EvidenceError(f"{label}: selected attention case is absent from the frozen contract")
    checks = result.get("checks")
    if not isinstance(checks, list) or [item.get("case") if isinstance(item, dict) else None
                                       for item in checks] != expected_ids:
        raise EvidenceError(f"{label}: native attention check list does not match the selected contract cases")

    numerical = contract.get("numerical_contract")
    if not isinstance(numerical, dict):
        raise EvidenceError(f"{label}: attention numerical contract is missing")
    output_atol, output_rtol = numerical.get("output_atol"), numerical.get("output_rtol")
    row_max_atol = numerical.get("row_max_atol")
    normalizer_atol, normalizer_rtol = numerical.get("normalizer_atol"), numerical.get("normalizer_rtol")
    tolerance_values = (output_atol, output_rtol, row_max_atol, normalizer_atol, normalizer_rtol)
    if any(not _attention_nonnegative(value) for value in tolerance_values):
        raise EvidenceError(f"{label}: attention tolerances are malformed")

    aggregate_outputs = aggregate_rows = 0
    error_maxima = {"max_abs_output_error": 0.0,
                    "max_row_max_abs_error": 0.0,
                    "max_row_normalizer_relative_error": 0.0}
    for item in checks:
        case_id = item["case"]
        dimensions = specs[case_id]
        if item.get("passed") is not True:
            raise EvidenceError(f"{label}: native attention check failed for {case_id}")
        for key, expected in (("checked_outputs", dimensions["outputs"]),
                              ("checked_rows", dimensions["rows"])):
            if type(item.get(key)) is not int or item[key] != expected:
                raise EvidenceError(f"{label}: native attention {case_id} {key} differs from the contract")
        numeric = ("max_abs_output_error", "max_row_max_abs_error",
                   "max_row_normalizer_relative_error")
        if any(not _attention_nonnegative(item.get(key)) for key in numeric):
            raise EvidenceError(f"{label}: native attention {case_id} has malformed error metrics")
        oracle = oracle_data.get(case_id)
        ref_values = oracle.get("output_fp64") if isinstance(oracle, dict) else None
        if not isinstance(ref_values, list) or len(ref_values) != dimensions["outputs"]:
            raise EvidenceError(f"{label}: independent FP64 outputs are missing for {case_id}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(float(value)) for value in ref_values):
            raise EvidenceError(f"{label}: independent FP64 outputs are malformed for {case_id}")
        max_reference = max((abs(float(value)) for value in ref_values), default=0.0)
        if float(item["max_abs_output_error"]) > float(output_atol) + float(output_rtol) * max_reference:
            raise EvidenceError(f"{label}: native attention output error exceeds the frozen envelope for {case_id}")
        if float(item["max_row_max_abs_error"]) > float(row_max_atol):
            raise EvidenceError(f"{label}: native attention row maximum exceeds tolerance for {case_id}")
        reference_rows = oracle.get("rows")
        if not isinstance(reference_rows, list) or len(reference_rows) != dimensions["rows"]:
            raise EvidenceError(f"{label}: independent FP64 row oracle is missing for {case_id}")
        relative_envelope = 0.0
        for row in reference_rows:
            if not isinstance(row, dict):
                raise EvidenceError(f"{label}: independent FP64 row oracle is malformed for {case_id}")
            valid_keys = row.get("valid_keys")
            normalizer = row.get("normalizer")
            if (type(valid_keys) is not int or valid_keys < 0 or
                    not _finite_nonnegative_number(normalizer)):
                raise EvidenceError(f"{label}: independent FP64 row oracle is malformed for {case_id}")
            if valid_keys == 0:
                if float(normalizer) != 0.0:
                    raise EvidenceError(f"{label}: empty-row FP64 normalizer is nonzero for {case_id}")
                continue
            if float(normalizer) <= 0.0:
                raise EvidenceError(f"{label}: nonempty FP64 row normalizer is not positive for {case_id}")
            relative_envelope = max(
                relative_envelope,
                float(normalizer_rtol) + float(normalizer_atol) / float(normalizer),
            )
        if float(item["max_row_normalizer_relative_error"]) > relative_envelope:
            raise EvidenceError(f"{label}: native attention normalizer exceeds tolerance for {case_id}")
        aggregate_outputs += dimensions["outputs"]
        aggregate_rows += dimensions["rows"]
        for key in error_maxima:
            error_maxima[key] = max(error_maxima[key], float(item[key]))

    errors = result.get("errors")
    if (not isinstance(errors, dict) or type(errors.get("checked_outputs")) is not int or
            errors.get("checked_outputs") != aggregate_outputs or
            type(errors.get("checked_rows")) is not int or errors.get("checked_rows") != aggregate_rows):
        raise EvidenceError(f"{label}: native attention aggregate counts are inconsistent")
    for key, maximum in error_maxima.items():
        if (not _attention_nonnegative(errors.get(key)) or
                float(errors[key]) != maximum):
            raise EvidenceError(f"{label}: native attention aggregate {key} is inconsistent")
    if mode == "time":
        timing = checks[0]
        samples = timing.get("samples_us")
        scope = timing.get("timing_scope")
        if (not isinstance(samples, list) or len(samples) != 11 or
                any(not _finite_positive_number(value) for value in samples) or
                timing.get("timing_phase") != "steady_state_compute" or
                not isinstance(scope, dict) or
                not isinstance(scope.get("included"), str) or
                "full selected implementation" not in scope["included"] or
                not isinstance(scope.get("excluded"), str) or
                "input loading" not in scope["excluded"] or
                "H2D upload" not in scope["excluded"]):
            raise EvidenceError(f"{label}: native attention timing samples or scope are incomplete")


def _attention_selector(argv: list[Any], role: str, flag: str, expected: str | None) -> None:
    matches = [index for index, value in enumerate(argv)
               if isinstance(value, str) and (value == flag or value.startswith(flag + "="))]
    expected_count = 0 if expected is None else 1
    if len(matches) != expected_count:
        raise EvidenceError(f"{role}: argv must contain {expected_count} {flag} selector(s)")
    if not matches:
        return
    index = matches[0]
    if (argv[index] != flag or index + 1 >= len(argv) or
            not isinstance(argv[index + 1], str) or argv[index + 1].startswith("--") or
            argv[index + 1] != expected):
        raise EvidenceError(f"{role}: argv {flag} selector differs from the attention role")


def _attention_validate_argv(summary: dict[str, Any], root: Path, role: str,
                             expected_impl: str, expected_mode: str, expected_case: str,
                             expected_splits: int,
                             implementation: dict[str, Any], input_path: Path,
                             oracle_path: Path) -> None:
    provenance = summary.get("provenance")
    argv = provenance.get("argv") if isinstance(provenance, dict) else None
    if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
        raise EvidenceError(f"{role}: controller provenance argv is missing")
    executable = Path(argv[0])
    if not executable.is_absolute():
        executable = root / executable
    binary = provenance.get("binary")
    binary_path = Path(str(binary.get("path", ""))) if isinstance(binary, dict) else Path()
    if not binary_path.is_absolute():
        binary_path = root / binary_path
    if (executable.resolve() != implementation["executed_path"].resolve() or
            binary_path.resolve() != implementation["executed_path"].resolve()):
        raise EvidenceError(f"{role}: controller argv executable differs from the declared attention binary")
    _attention_selector(argv, role, "--input", str(input_path.relative_to(root)))
    _attention_selector(argv, role, "--reference", str(oracle_path.relative_to(root)))
    _attention_selector(argv, role, "--impl", expected_impl)
    _attention_selector(argv, role, "--mode", expected_mode)
    _attention_selector(argv, role, "--case", expected_case)
    if expected_mode == "time":
        _attention_selector(argv, role, "--warmups", "2")
        _attention_selector(argv, role, "--repetitions", "11")
    else:
        _attention_selector(argv, role, "--warmups", None)
        _attention_selector(argv, role, "--repetitions", None)
    if expected_impl == "split":
        _attention_selector(argv, role, "--splits", str(expected_splits))
    else:
        _attention_selector(argv, role, "--splits", None)


def _attention_validate_profile(root: Path, base: Path, run: dict[str, Any],
                                summary: dict[str, Any], label: str) -> None:
    profile = run.get("profile")
    if not isinstance(profile, dict):
        raise EvidenceError(f"{label}: NCU profile artifacts are missing")
    report = checked_file(base, profile.get("report"), f"{label} NCU report")
    analysis_path = checked_file(base, profile.get("analysis"), f"{label} NCU analysis")
    shim_log = checked_file(base, profile.get("shim_log"), f"{label} NCU shim log")
    analysis = read_json(analysis_path)
    selected_kernel = profile.get("selected_kernel")
    hot_kernel = analysis.get("hot_kernel") if isinstance(analysis, dict) else None
    kernel_name = hot_kernel.get("name") if isinstance(hot_kernel, dict) else None
    if (not isinstance(selected_kernel, str) or not selected_kernel or
            not isinstance(kernel_name, str) or
            not (kernel_name == selected_kernel or kernel_name.startswith(selected_kernel + "(")) or
            analysis.get("counter_valid") is not True or analysis.get("timing_valid") is not False or
            analysis.get("measurement_scope") != "kernel counters" or not report.stat().st_size):
        raise EvidenceError(f"{label}: NCU report or parsed summary does not bind a valid selected kernel/counter profile")
    provenance = summary.get("provenance")
    tools = provenance.get("tools") if isinstance(provenance, dict) else None
    ncu_identity = tools.get("ncu") if isinstance(tools, dict) else None
    if (not isinstance(ncu_identity, dict) or not ncu_identity.get("path") or
            not ncu_identity.get("version")):
        raise EvidenceError(f"{label}: NCU tool identity is missing")
    try:
        shim_records = [json.loads(line) for line in shim_log.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label}: malformed NCU shim invocation log") from exc
    shim_path = absolute_repo_file(root, "benchmark_runs/volta-environment/ncu-collection-shim.py",
                                   f"{label} checked-in NCU shim")
    shim_sha = digest(shim_path)
    ncu_path = Path(str(ncu_identity["path"])).resolve()
    inputs = _input_bindings(summary)
    collections = [item for item in shim_records if isinstance(item, dict) and
                   item.get("policy") == "collection-clock-control-none-source-import-yes"]
    if not collections or not all(
        isinstance(item.get("argv_executed"), list) and
        _argument_values(item["argv_executed"], "--clock-control") == ["none"] and
        _argument_values(item["argv_executed"], "--import-source") == ["yes"] and
        isinstance(item.get("shim_path"), str) and Path(item["shim_path"]).resolve() == shim_path and
        item.get("shim_sha256") == shim_sha and inputs.get(str(shim_path)) == shim_sha and
        isinstance(item.get("actual_ncu_path"), str) and
        Path(item["actual_ncu_path"]).resolve() == ncu_path and ncu_path.is_file() and
        item.get("actual_ncu_sha256") == digest(ncu_path) and
        inputs.get(str(ncu_path)) == item.get("actual_ncu_sha256")
        for item in collections
    ):
        raise EvidenceError(f"{label}: NCU collection does not bind the no-clock/source-import policy")


def _attention_archive_manifest(base: Path, run: dict[str, Any], role: str,
                                evidence_id: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = checked_file(base, run.get("archive_manifest"),
                                 f"attention {role} archive manifest")
    manifest = read_json(manifest_path)
    archive_role = ATTENTION_ARCHIVE_ROLE_ALIASES.get(role, role)
    if (not isinstance(manifest, dict) or
            manifest.get("format") != "KERNELWIKI-ATTENTION-ROLE-ARCHIVE/1" or
            manifest.get("role") != archive_role or manifest.get("evidence_id") != evidence_id or
            not isinstance(manifest.get("files"), dict)):
        raise EvidenceError(f"attention {role}: archive manifest identity is invalid")
    return manifest_path, manifest


def _attention_bind_stdout_capture(base: Path, root: Path, run: dict[str, Any],
                                   role: str, native_stdout: Path,
                                   sanitizer_summary: Path | None = None,
                                   sanitizer_log: Path | None = None) -> None:
    """Bind app stdout to the archive, and bind outer stdout to the PC export.

    Direct runner output is itself the controller-captured stdout artifact. For
    sanitizer and NCU, Project Control captures the wrapper's outer stdout;
    their inner application stdout is separately archived and its reported
    runtime path must be named by that outer text.
    """
    evidence_record = run.get("evidence")
    if not isinstance(evidence_record, dict):
        raise EvidenceError(f"attention {role}: exported controller evidence is missing")
    evidence_path = checked_file(base, evidence_record, f"attention {role} evidence")
    exported = read_json(evidence_path)
    evidence = exported.get("evidence") if isinstance(exported, dict) else None
    evidence_id = evidence.get("id") if isinstance(evidence, dict) else None
    if not isinstance(evidence_id, str) or not evidence_id:
        raise EvidenceError(f"attention {role}: exported controller evidence ID is missing")
    manifest_path, manifest = _attention_archive_manifest(base, run, role, evidence_id)
    files = manifest["files"]

    def archive_hash(path: Path, label: str) -> str:
        try:
            archive_key = path.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
        except ValueError as exc:
            raise EvidenceError(f"attention {role}: {label} is outside its role archive") from exc
        expected = files.get(archive_key)
        if not isinstance(expected, str) or expected != digest(path):
            raise EvidenceError(f"attention {role}: {label} differs from the archive manifest")
        return expected

    execution_path = checked_file(base, run.get("execution"), f"attention {role} execution")
    lease_record = run.get("lease")
    if not isinstance(lease_record, dict) or not isinstance(lease_record.get("path"), str):
        raise EvidenceError(f"attention {role}: archived lease path is missing")
    lease_path = (root / lease_record["path"]).resolve()
    # The checked controller lease is outside the evidence base by design.
    if (not lease_path.is_relative_to(root.resolve()) or not lease_path.is_file() or
            digest(lease_path) != lease_record.get("sha256")):
        raise EvidenceError(f"attention {role}: archived lease does not match the run record")
    expected_archive_files = {
        "project-control-result.json": evidence_path,
        "controller-run.json": execution_path,
        "lease.json": lease_path,
    }
    for archive_key, path in expected_archive_files.items():
        expected = files.get(archive_key)
        if not isinstance(expected, str) or expected != digest(path):
            raise EvidenceError(f"attention {role}: archive manifest does not bind {archive_key}")

    evidence_summary = evidence.get("summary") if isinstance(evidence, dict) else None
    record = evidence_summary.get("record") if isinstance(evidence_summary, dict) else None
    outer_stdout_path = record.get("stdout") if isinstance(record, dict) else None
    artifacts = evidence.get("artifacts") if isinstance(evidence, dict) else None
    stdout_artifacts = [item for item in artifacts if isinstance(item, dict) and
                        item.get("kind") == "stdout"] if isinstance(artifacts, list) else []
    nested_capture = role in {"sanitizer_candidate", "sanitizer_split_decode",
                              "profile_candidate_decode"}
    outer_archive_name = "controller.stdout.txt" if nested_capture else "runner.stdout.txt"
    outer_stdout_hash = files.get(outer_archive_name)
    if (not isinstance(outer_stdout_path, str) or len(stdout_artifacts) != 1 or
            stdout_artifacts[0].get("complete") != 1 or
            stdout_artifacts[0].get("path") != outer_stdout_path or
            stdout_artifacts[0].get("content_hash") != outer_stdout_hash or
            not isinstance(outer_stdout_hash, str) or len(outer_stdout_hash) != 64):
        raise EvidenceError(f"attention {role}: controller stdout artifact is not hash-bound to its export")
    outer_stdout = manifest_path.parent / outer_archive_name
    if not outer_stdout.is_file() or digest(outer_stdout) != outer_stdout_hash:
        raise EvidenceError(f"attention {role}: archived controller stdout differs from its exported artifact")
    try:
        outer_text = outer_stdout.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise EvidenceError(f"attention {role}: archived controller stdout cannot be read: {exc}") from exc

    capture = run.get("native_stdout_capture")
    native_record = checked_file(base, capture, f"attention {role} native stdout capture")
    if native_record.resolve() != native_stdout.resolve():
        raise EvidenceError(f"attention {role}: native stdout path differs from its declared capture")
    native_hash = archive_hash(native_record, "native stdout")
    if not isinstance(capture, dict) or capture.get("sha256") != native_hash:
        raise EvidenceError(f"attention {role}: native stdout archive hash differs from the capture record")

    if not nested_capture:
        try:
            native_suffix = native_record.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
        except ValueError as exc:
            raise EvidenceError(f"attention {role}: direct stdout is outside its role archive") from exc
        if (native_suffix != "runner.stdout.txt" or capture.get("reported_path") is not None or
                native_hash != outer_stdout_hash or
                stdout_artifacts[0].get("content_hash") != native_hash):
            raise EvidenceError(f"attention {role}: native stdout is not the captured controller stdout")
        return

    expected_suffix = ("compute-sanitizer/stdout.txt" if role.startswith("sanitizer_")
                       else "ncu/command.stdout.txt")
    try:
        suffix = native_record.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"attention {role}: nested stdout capture is outside its role archive") from exc
    reported_path = capture.get("reported_path") if isinstance(capture, dict) else None
    if (suffix != expected_suffix or not isinstance(reported_path, str) or
            not Path(reported_path).is_absolute() or reported_path not in outer_text):
        raise EvidenceError(f"attention {role}: nested stdout path is not identified by captured outer output")

    if role.startswith("sanitizer_"):
        if sanitizer_log is None or sanitizer_summary is None:
            raise EvidenceError(f"attention {role}: sanitizer raw log and summary bindings are missing")
        log_hash = archive_hash(sanitizer_log, "raw Compute Sanitizer log")
        summary_hash = archive_hash(sanitizer_summary, "Compute Sanitizer classifier summary")
        log_record = run.get("sanitizer_log")
        summary_record = run.get("sanitizer_summary")
        if (files.get("compute-sanitizer/summary.json") != summary_hash or
                files.get("compute-sanitizer/raw.log") != log_hash or
                not isinstance(log_record, dict) or
                log_hash != log_record.get("sha256") or
                not isinstance(summary_record, dict) or
                summary_hash != summary_record.get("sha256")):
            raise EvidenceError(f"attention {role}: sanitizer raw log/summary is not bound to its archive")
        if f"stdout: {reported_path}" not in outer_text:
            raise EvidenceError(f"attention {role}: captured sanitizer output does not identify native stdout")
        raw_record = log_record
        if not isinstance(raw_record, dict) or not isinstance(raw_record.get("reported_path"), str):
            raise EvidenceError(f"attention {role}: raw sanitizer log reported path is missing")
        if f"raw log: {raw_record['reported_path']}" not in outer_text:
            raise EvidenceError(f"attention {role}: captured sanitizer output does not identify the raw log")
    else:
        # The NCU outer report names the command stdout path; the archived
        # native JSON is that separately captured file, not PC's outer stdout.
        if f"command stdout: {reported_path}" not in outer_text:
            raise EvidenceError(f"attention {role}: captured NCU output does not identify command stdout")


def _attention_source_closure(root: Path, case: dict[str, Any],
                              captured_archives: dict[str, dict[str, Any]]) -> Path:
    closure_path, _closure_sha = sha_binding(root, case.get("source_closure"),
                                             "attention v1 source closure")
    closure = read_json(closure_path)
    records = closure.get("repository_inputs") if isinstance(closure, dict) else None
    if (not isinstance(closure, dict) or
            closure.get("format") != "KERNELWIKI-ATTENTION-V1-SOURCE-CLOSURE/1" or
            not isinstance(records, list) or not records):
        raise EvidenceError("attention v1 source closure is malformed")
    expected: dict[str, tuple[str, int]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise EvidenceError("attention v1 source closure has a malformed repository input")
        original = _repo_relative_identity(root, item.get("original_path"),
                                           "attention v1 captured original path")
        archived = item.get("archived_path")
        archive_file = absolute_repo_file(root, archived, "attention v1 source closure archive")
        sha = item.get("sha256")
        size = item.get("size_bytes")
        if (not isinstance(sha, str) or len(sha) != 64 or digest(archive_file) != sha or
                type(size) is not int or archive_file.stat().st_size != size):
            raise EvidenceError("attention v1 source closure archive hash or size differs")
        key = str(original)
        if key in expected:
            raise EvidenceError("attention v1 source closure repeats an original input path")
        expected[key] = (str(archive_file), size)
    if set(expected) != set(captured_archives):
        raise EvidenceError("attention captured input archives differ from the v1 source closure")
    for original, (archived, _size) in expected.items():
        capture = captured_archives[original]
        if (str(capture["archive_path"]) != archived or
                capture["sha256"] != digest(Path(archived))):
            raise EvidenceError("attention captured input archive does not match the v1 closure manifest")
    return closure_path


def _attention_role_source_closure(root: Path, run: dict[str, Any], role: str,
                                  captured_archives: dict[str, dict[str, Any]]) -> Path:
    closure_path, _closure_sha = sha_binding(root, run.get("source_closure"),
                                             f"attention {role} v2 source closure")
    closure = read_json(closure_path)
    records = closure.get("repository_inputs") if isinstance(closure, dict) else None
    if (not isinstance(closure, dict) or
            closure.get("format") != "KERNELWIKI-ATTENTION-V2-SOURCE-CLOSURE/1" or
            not isinstance(records, list) or not records):
        raise EvidenceError(f"attention {role}: v2 source closure is malformed")
    expected: dict[str, tuple[str, str, int]] = {}
    for index, item in enumerate(records):
        label = f"attention {role} v2 source closure input {index}"
        if not isinstance(item, dict):
            raise EvidenceError(f"{label}: input record is malformed")
        original = _repo_relative_identity(root, item.get("original_path"),
                                           f"{label} original path")
        archive_raw, expected_sha, size = (item.get("archived_path"), item.get("sha256"),
                                           item.get("size_bytes"))
        if not isinstance(archive_raw, str):
            raise EvidenceError(f"{label}: archive path is malformed")
        archive_path = absolute_repo_file(root, archive_raw, f"{label} archive")
        if (not isinstance(expected_sha, str) or len(expected_sha) != 64 or
                digest(archive_path) != expected_sha or type(size) is not int or
                archive_path.stat().st_size != size or str(original) in expected):
            raise EvidenceError(f"{label}: archive hash, size, or identity is invalid")
        expected[str(original)] = (str(archive_path), expected_sha, size)
    if set(expected) != set(captured_archives):
        raise EvidenceError(f"attention {role}: captured aliases differ from its v2 source closure")
    for original, (archive, expected_sha, size) in expected.items():
        capture = captured_archives[original]
        if (str(capture["archive_path"]) != archive or capture["sha256"] != expected_sha or
                capture["archive_path"].stat().st_size != size):
            raise EvidenceError(f"attention {role}: captured alias differs from its v2 closure identity")

    declaring_specs = closure.get("declaring_specs")
    expected_specs = {
        "benchmark_runs/volta-attention/controller/v2/role-attention-correctness_split16.spec.json",
        "benchmark_runs/volta-attention/controller/v2/role-attention-sanitizer_split16.spec.json",
        "benchmark_runs/volta-attention/controller/v2/role-attention-split16_decode_timing.spec.json",
    }
    if (not isinstance(declaring_specs, list) or
            {item.get("path") for item in declaring_specs if isinstance(item, dict)} != expected_specs or
            len(declaring_specs) != len(expected_specs)):
        raise EvidenceError(f"attention {role}: v2 source closure spec set is invalid")
    for spec in declaring_specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("sha256"), str):
            raise EvidenceError(f"attention {role}: v2 source closure spec binding is malformed")
        original = _repo_relative_identity(root, spec.get("path"),
                                           f"attention {role} v2 spec path")
        capture = captured_archives.get(str(original))
        if capture is None or capture["sha256"] != spec["sha256"]:
            raise EvidenceError(f"attention {role}: v2 source closure does not archive its declaring spec")
        spec_data = read_json(capture["archive_path"])
        inputs = spec_data.get("inputs") if isinstance(spec_data, dict) else None
        if not isinstance(inputs, list):
            raise EvidenceError(f"attention {role}: archived v2 spec has no declared inputs")
        for item in inputs:
            if not isinstance(item, str) or not item:
                raise EvidenceError(f"attention {role}: archived v2 spec input list is malformed")
            candidate = Path(item)
            if candidate.is_absolute():
                continue
            declared = _repo_relative_identity(root, item, f"attention {role} v2 declared input")
            if str(declared) not in captured_archives:
                raise EvidenceError(f"attention {role}: v2 declared repository input is absent from its closure")
    return closure_path


def _check_attention_case(base: Path, root: Path, case: dict[str, Any],
                          evidence_ids: set[str], contract_path: Path,
                          contract_sha: str, contract: dict[str, Any]) -> None:
    case_id = case["id"]
    if case_id != "attention" or contract.get("case_id") != "attention":
        raise EvidenceError("attention evidence must bind the canonical attention contract")
    input_path, input_sha = sha_binding(root, case.get("represented_input"),
                                        "attention represented input")
    oracle_path, oracle_sha = sha_binding(root, case.get("oracle"), "attention FP64 reference")
    case_index_path, case_index_sha = sha_binding(root, case.get("case_index"),
                                                  "attention case index")
    source_manifest_path, source_manifest_sha = sha_binding(root, case.get("source_manifest"),
                                                            "attention source manifest")
    contract_input = contract.get("represented_input")
    contract_oracle = contract.get("oracle")
    if (not isinstance(contract_input, dict) or not isinstance(contract_oracle, dict) or
            input_sha != contract_input.get("sha256") or
            oracle_sha != contract_oracle.get("sha256") or
            input_path.name != contract_input.get("path") or
            oracle_path.name != contract_oracle.get("path")):
        raise EvidenceError("attention bundle/reference differs from the frozen contract")
    case_specs = _attention_case_dimensions(contract)
    case_index = read_json(case_index_path)
    indexed_cases = case_index.get("cases") if isinstance(case_index, dict) else None
    if (not isinstance(case_index, dict) or
            case_index.get("format") != "KERNELWIKI-ATTENTION-CASES/1" or
            not isinstance(indexed_cases, list) or len(indexed_cases) != len(case_specs) or
            any(not isinstance(item, dict) or not isinstance(item.get("id"), str)
                for item in indexed_cases) or
            {item["id"] for item in indexed_cases} != set(case_specs)):
        raise EvidenceError("attention case index differs from the frozen seven-case contract")
    indexed_input = case_index.get("combined_input")
    indexed_reference = case_index.get("combined_reference")
    if (not isinstance(indexed_input, dict) or not isinstance(indexed_reference, dict) or
            indexed_input.get("sha256") != input_sha or
            indexed_reference.get("sha256") != oracle_sha):
        raise EvidenceError("attention case index does not bind the combined input and FP64 reference")

    case_oracles = case.get("case_oracles")
    if not isinstance(case_oracles, list) or len(case_oracles) != len(case_specs):
        raise EvidenceError("attention per-case FP64 oracle bindings are missing")
    oracle_records: dict[str, dict[str, Any]] = {}
    oracle_paths: dict[str, Path] = {}
    for item in case_oracles:
        if not isinstance(item, dict) or not isinstance(item.get("case_id"), str):
            raise EvidenceError("attention per-case FP64 oracle binding is malformed")
        per_case_id = item["case_id"]
        if per_case_id not in case_specs or per_case_id in oracle_records:
            raise EvidenceError("attention per-case FP64 oracle set is duplicated or unexpected")
        path, sha = sha_binding(root, item, f"attention {per_case_id} FP64 oracle")
        data = read_json(path)
        expected = case_specs[per_case_id]
        if (sha != next((entry.get("oracle_sha256") for entry in indexed_cases
                         if isinstance(entry, dict) and entry.get("id") == per_case_id), None) or
                data.get("format") != "KERNELWIKI-CPU-ORACLE/1" or
                data.get("passed") is not True or data.get("case") != per_case_id or
                data.get("shape") != expected["shape"] or
                not isinstance(data.get("output_fp64"), list) or
                len(data["output_fp64"]) != expected["outputs"] or
                not isinstance(data.get("rows"), list) or len(data["rows"]) != expected["rows"]):
            raise EvidenceError(f"attention {per_case_id} FP64 oracle differs from the frozen case contract")
        oracle_records[per_case_id] = data
        oracle_paths[per_case_id] = path
    if set(oracle_records) != set(case_specs):
        raise EvidenceError("attention per-case FP64 oracle set is incomplete")

    captured_archives = _captured_input_archives(root, case, required=True)
    _attention_source_closure(root, case, captured_archives)
    source_manifest_record = case.get("source_manifest")
    if isinstance(source_manifest_record, dict) and "original_path" in source_manifest_record:
        original_manifest = _repo_relative_identity(root, source_manifest_record.get("original_path"),
                                                    "attention source manifest original path")
        archive = captured_archives.get(str(original_manifest))
        if (archive is None or source_manifest_path != archive["archive_path"] or
                source_manifest_sha != archive["sha256"]):
            raise EvidenceError("attention source manifest does not match its v1 captured input")
    elif source_manifest_record is not None:
        raise EvidenceError("attention source manifest must identify its v1 captured input")
    additional_inputs: dict[str, str] = {
        str(case_index_path.resolve()): case_index_sha,
    }
    for item in indexed_cases:
        contract_case = case_specs[item["id"]]["signature"]
        if (item.get("shape") != contract_case.get("q_shape") or
                item.get("kv_shape") != contract_case.get("kv_shape") or
                any(item.get(field) != contract_case.get(field)
                    for field in ("causal", "head_dim", "block_size",
                                  "has_per_query_boolean_mask", "tolerance_group"))):
            raise EvidenceError(f"attention case index metadata differs from contract for {item['id']}")
        if not isinstance(item, dict):
            raise EvidenceError("attention case index contains a malformed input entry")
        for field, hash_field in (("input", "input_sha256"),
                                  ("metadata", "metadata_sha256"),
                                  ("oracle", "oracle_sha256")):
            raw = item.get(field)
            expected_sha = item.get(hash_field)
            if (not isinstance(raw, str) or not isinstance(expected_sha, str) or
                    len(expected_sha) != 64):
                raise EvidenceError(f"attention case index {field} identity is malformed")
            path = (case_index_path.parent / raw).resolve()
            if (not path.is_relative_to(root) or not path.is_file() or digest(path) != expected_sha):
                raise EvidenceError(f"attention case index {field} is missing or has a SHA-256 mismatch")
            additional_inputs[str(path)] = expected_sha
        if str(oracle_paths[item["id"]].resolve()) != str((case_index_path.parent / item["oracle"]).resolve()):
            raise EvidenceError(f"attention {item['id']} FP64 oracle path differs from the case index")

    captured_inputs = {str(item["original_path"]): item["sha256"]
                       for item in captured_archives.values()}
    implementations = case.get("implementations")
    if not isinstance(implementations, dict):
        raise EvidenceError("attention implementations are missing")
    expected_impl_names = {item[0] for item in ATTENTION_REQUIRED_ROLES.values()}
    runs = case.get("runs")
    if not isinstance(runs, dict):
        raise EvidenceError("attention controller runs are missing")
    present_optional = set(runs) & set(ATTENTION_OPTIONAL_ROLES)
    if present_optional and present_optional != set(ATTENTION_OPTIONAL_ROLES):
        raise EvidenceError("attention split evidence must include correctness, sanitizer, and timing roles together")
    if set(runs) - set(ATTENTION_REQUIRED_ROLES) - set(ATTENTION_OPTIONAL_ROLES):
        raise EvidenceError("attention controller runs contain an unknown role")
    if not set(ATTENTION_REQUIRED_ROLES).issubset(runs):
        raise EvidenceError("attention controller runs are missing a required role")
    if present_optional:
        expected_impl_names.add("split")
    if not expected_impl_names.issubset(implementations):
        raise EvidenceError("attention implementation identity is missing for a measured role")

    bound_impls: dict[str, dict[str, Any]] = {}
    for name in expected_impl_names:
        entry = implementations.get(name)
        if not isinstance(entry, dict):
            raise EvidenceError(f"attention {name} implementation is malformed")
        source_path, source_sha = sha_binding(root, entry.get("source"), f"attention {name} source")
        source_input_path = source_path
        source_original_path: Path | None = None
        source_record = entry.get("source")
        if isinstance(source_record, dict) and "original_path" in source_record:
            source_original_path = _repo_relative_identity(
                root, source_record.get("original_path"),
                f"attention {name} source original path")
            source_input_path = source_original_path
        else:
            raise EvidenceError(f"attention {name} source lacks its original repository identity")
        binary_path, binary_sha = sha_binding(root, entry.get("binary"), f"attention {name} binary")
        binary_record = entry.get("binary")
        executed_raw = entry.get("executed_path", binary_record.get("path") if isinstance(binary_record, dict) else None)
        executed_path = absolute_repo_file(root, executed_raw, f"attention {name} executed binary")
        bound_impls[name] = {"source_path": source_path, "source_input_path": source_input_path,
                             "source_original_path": source_original_path,
                             "source_sha": source_sha, "binary_path": binary_path,
                             "executed_path": executed_path, "binary_sha": binary_sha}

    libraries = case.get("libraries")
    if not isinstance(libraries, list):
        raise EvidenceError("attention linked library identity must be an explicit list")
    bound_libraries = []
    for item in libraries:
        if (not isinstance(item, dict) or item.get("implementation") not in expected_impl_names or
                not all(item.get(key) for key in ("name", "version", "path", "sha256"))):
            raise EvidenceError("attention linked library entry lacks identity/version/path/hash")
        raw_path = Path(str(item["path"]))
        library_path = (raw_path if raw_path.is_absolute() else root / raw_path).resolve()
        if not library_path.is_file() or digest(library_path) != item["sha256"]:
            raise EvidenceError("attention linked library path/hash is unavailable or mismatched")
        bound_libraries.append((item, library_path))

    summaries: dict[str, dict[str, Any]] = {}
    role_specs = dict(ATTENTION_REQUIRED_ROLES)
    role_specs.update({key: value for key, value in ATTENTION_OPTIONAL_ROLES.items() if key in runs})
    native_outputs: dict[str, Path] = {}
    for role, (impl_name, native_impl, kind, mode, selected_case) in role_specs.items():
        if impl_name not in bound_impls:
            raise EvidenceError(f"attention {role}: implementation is not archived")
        role_archives = captured_archives
        if role in ATTENTION_OPTIONAL_ROLES:
            role_archives = _captured_input_archives(
                root, {"id": role, "captured_input_archives": runs[role].get("captured_input_archives")},
                required=True)
            _attention_role_source_closure(root, runs[role], role, role_archives)
        role_captured_inputs = {str(item["original_path"]): item["sha256"]
                                for item in role_archives.values()}
        allow_duplicate_captured_inputs = {
            str(path.resolve())
            for path in (contract_path, input_path, oracle_path, case_index_path)
        }
        allow_duplicate_captured_inputs.update(
            str(archive["original_path"])
            for archive in role_archives.values()
            if archive["original_path"].name == "source-manifest.json" and
            "cases" in archive["original_path"].parts and
            "attention" in archive["original_path"].parts
        )
        source_original = bound_impls[impl_name]["source_original_path"]
        if source_original is not None:
            source_capture = role_archives.get(str(source_original))
            if (source_capture is None or
                    bound_impls[impl_name]["source_path"] != source_capture["archive_path"] or
                    bound_impls[impl_name]["source_sha"] != source_capture["sha256"]):
                raise EvidenceError(f"attention {role}: implementation source differs from its role archive")
        summary = _controller_run(
            base, root, runs.get(role), f"attention {role}", kind, bound_impls[impl_name],
            contract_path, contract_sha, input_path, input_sha, oracle_path, oracle_sha,
            str(case.get("gpu_uuid", "")), evidence_ids, role_captured_inputs,
            additional_inputs=additional_inputs,
            allow_duplicate_captured_inputs=allow_duplicate_captured_inputs,
        )
        if role in ATTENTION_OPTIONAL_ROLES:
            argv_summ = summary.get("provenance", {}).get("argv", [])
            split_values = _argument_values(argv_summ, "--splits")
            if len(split_values) != 1 or split_values[0] != "16":
                raise EvidenceError(f"attention {role}: supplemental split role must use exactly 16 splits")
            split_count = int(split_values[0])
        else:
            split_count = 1
        _attention_validate_argv(summary, root, role, native_impl, mode, selected_case,
                                 split_count, bound_impls[impl_name], input_path, oracle_path)
        native_path = checked_file(base, runs[role].get("native_stdout"),
                                   f"attention {role} native stdout")
        native_outputs[role] = native_path
        sanitizer_summary_path = None
        sanitizer_log_path = None
        if role in {"sanitizer_candidate", "sanitizer_split_decode"}:
            sanitizer_summary_path = checked_file(
                base, runs[role].get("sanitizer_summary"),
                f"attention {role} sanitizer classifier summary")
            sanitizer_log_path = checked_file(
                base, runs[role].get("sanitizer_log"),
                f"attention {role} raw Compute Sanitizer log")
        _attention_bind_stdout_capture(base, root, runs[role], role, native_path,
                                       sanitizer_summary_path, sanitizer_log_path)
        native_result = read_json(native_path)
        _attention_native_checks(native_result, contract, oracle_records,
                                 str(case.get("gpu_uuid", "")), native_impl, mode,
                                 selected_case, split_count, f"attention {role}")
        summaries[role] = summary

    for library, library_path in bound_libraries:
        implementation = library["implementation"]
        roles = [role for role, (impl_name, _native_impl, _kind, _mode, _case) in role_specs.items()
                 if impl_name == implementation]
        for role in roles:
            if _input_bindings(summaries[role]).get(str(library_path)) != library["sha256"]:
                raise EvidenceError(f"attention {role}: controller inputs do not bind {library['name']}")

    sanitizer_roles = ["sanitizer_candidate"]
    if "sanitizer_split_decode" in summaries:
        sanitizer_roles.append("sanitizer_split_decode")
    for role in sanitizer_roles:
        sanitizer = checked_file(base, runs[role].get("sanitizer_summary"),
                                 f"attention {role} sanitizer classifier summary")
        sanitizer_data = read_json(sanitizer)
        if (not isinstance(sanitizer_data, dict) or
                sanitizer_data.get("tool") != "compute-sanitizer" or
                sanitizer_data.get("status") != "partial" or
                sanitizer_data.get("tool_detail") != "memcheck" or
                sanitizer_data.get("conclusive") is not False or
                type(sanitizer_data.get("exit_status")) is not int or sanitizer_data["exit_status"] != 0 or
                type(sanitizer_data.get("error_count")) is not int or sanitizer_data["error_count"] != 0):
            raise EvidenceError(f"attention {role} sanitizer classifier is not the accepted raw-zero partial memcheck result")
        sanitizer_log = checked_file(base, runs[role].get("sanitizer_log"),
                                     f"attention {role} raw Compute Sanitizer log")
        try:
            log_text = sanitizer_log.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise EvidenceError(f"attention {role} raw Compute Sanitizer log cannot be read: {exc}") from exc
        summary_lines = [line for line in log_text.splitlines() if "ERROR SUMMARY:" in line]
        if len(summary_lines) != 1 or not any(
                line.strip().endswith("0 errors") for line in summary_lines):
            raise EvidenceError(f"attention {role} raw Compute Sanitizer log does not show exactly zero errors")

    if "profile_candidate_decode" not in summaries:
        raise EvidenceError("attention NCU profile role is missing")
    _attention_validate_profile(root, base, runs["profile_candidate_decode"],
                                summaries["profile_candidate_decode"],
                                "attention profile_candidate_decode")


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
    if _is_attention_contract(contract_data):
        _check_attention_case(base, root, case, evidence_ids, contract, contract_sha, contract_data)
        return
    rows_contract = _is_rows_contract(contract_data)
    captured_archives = _captured_input_archives(root, case, required=rows_contract)
    if rows_contract and {archive["original_path"].relative_to(root).as_posix()
                          for archive in captured_archives.values()} != ROW_CAPTURED_SOURCE_PATHS:
        raise EvidenceError(f"{case['id']}: captured source archive set does not match the frozen four-file snapshot")
    captured_inputs = {str(archive["original_path"]): archive["sha256"]
                       for archive in captured_archives.values()}
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
        source_record = entry.get("source")
        source_path, source_sha = sha_binding(root, source_record, f"{case['id']} {name} source")
        source_input_path = source_path
        if isinstance(source_record, dict) and "original_path" in source_record:
            original_path = _repo_relative_identity(
                root, source_record.get("original_path"), f"{case['id']} {name} source original path"
            )
            archive = captured_archives.get(str(original_path))
            if (archive is None or source_path != archive["archive_path"] or
                    source_sha != archive["sha256"]):
                raise EvidenceError(f"{case['id']} {name} source does not match its captured input archive")
            source_input_path = original_path
        elif rows_contract:
            raise EvidenceError(f"{case['id']} {name} source is missing its captured original_path")
        binary_path, binary_sha = sha_binding(root, entry.get("binary"), f"{case['id']} {name} binary")
        binary_record = entry.get("binary")
        default_executed = binary_record.get("path") if isinstance(binary_record, dict) else None
        executed_raw = entry.get("executed_path", default_executed)
        executed_path = absolute_repo_file(root, executed_raw, f"{case['id']} {name} executed binary")
        bound_impls[name] = {"source_path": source_path, "source_input_path": source_input_path,
                             "source_sha": source_sha,
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
            str(case.get("gpu_uuid", "")), evidence_ids, captured_inputs,
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
    if not _is_rows_contract(contract_data):
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
    shim_relative_path = ("benchmark_runs/volta-rows/controller/ncu-collection-shim.py"
                          if _is_rows_contract(contract_data) else
                          "benchmark_runs/volta-environment/ncu-collection-shim.py")
    shim_path = absolute_repo_file(root, shim_relative_path,
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
        (_is_rows_contract(contract_data) or
         profile_inputs.get(str(ncu_path)) == item.get("actual_ncu_sha256"))
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
