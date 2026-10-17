#!/usr/bin/env python3
"""Fail-closed checks for source-bound, controller-backed Volta comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
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
KWV07_SUITE_CONTRACT_FORMAT = "KERNELWIKI-KWV07-SUITE-CONTRACT/1"
KWV07_SUITE_FORMAT = "KERNELWIKI-KWV07-SUITE/1"
KWV07_NATIVE_FORMAT = "KERNELWIKI-KWV07-NATIVE/1"
KWV08_CONTRACT_FORMAT = "KERNELWIKI-VOLTA-RECURRENCE-CONTRACT/1"
KWV08_NATIVE_FORMAT = "KERNELWIKI-VOLTA-RECURRENCE-RUN/1"
KWV08_SOURCE_CLOSURE_FORMAT = "KERNELWIKI-KWV08-SOURCE-CLOSURE/1"
KWV08_INPUT_SHA256 = "e760eda922a42c0d6f6338e8cc0d0b0c7971c4205c956047a55791191fced4b5"
KWV08_PAYLOAD_SHA256 = "7c26087a2d035d692c90167cfc493756a3954cee84664a92c7ece2641611da80"
KWV08_SOURCE_CLOSURE_SHA256 = "fecce508e9e8e80567379b52b8a6bb88ea7c0a0f15222f711edd60a0e9598543"
KWV08_ROLES = {
    "correctness_baseline": ("foreground-baseline", "baseline", "per_token", "correctness", None),
    "correctness_candidate": ("foreground-baseline", "candidate", "retained", "correctness", None),
    "sanitizer_candidate": ("foreground-compute-sanitizer", "candidate", "retained", "correctness", None),
    "baseline_timing": ("foreground-baseline", "baseline", "per_token", "time", None),
    "candidate_timing": ("foreground-baseline", "candidate", "retained", "time", None),
    "profile_candidate": ("foreground-ncu", "candidate", "retained", "profile", "long_prefill_many_sequences"),
}
KWV08_CHECKS = {
    "full_sequence_all_outputs": "outputs",
    "full_sequence_padding_untouched": "padding",
    "full_sequence_final_state": "state",
    "segmented_carry_all_outputs": "outputs",
    "segmented_padding_untouched": "padding",
    "segmented_carry_final_state": "state",
}
KWV07_FROZEN_BUNDLE_SHA256 = "646137b453a81b9a09245d2b768376fab428ab4b4123c8023973d1b30942d3d8"
KWV07_FROZEN_SUITE_FILES_SHA256 = "c3de6df5f58c575eef30c565085b98dc07db6e413cb49eb52746d59c60e08e6e"
KWV07_FROZEN_SUITES = {
    "kwv07-grouped-gated": {
        "suite_sha256": "a1fcbe536ca98b14285f11af1b19d71302ccd46ed47bc8760d400fc092672b99",
        "contract_sha256": "ba610f656dd070d9e3bbe732677e9650beb9b69824c947e0341c8973d1a794d9",
        "oracle_sha256": "593c62facc7ffcba0ece321fe759f17b1e3750bfcf8eb640410653b96fcf8e67",
        "cases": {
            "tiny_balanced": {"shape": {"source_rows": 4, "assignments": 8, "k": 64, "n": 32, "experts": 4}, "outputs": 256},
            "tiny_skewed": {"shape": {"source_rows": 4, "assignments": 8, "k": 64, "n": 32, "experts": 4}, "outputs": 256},
            "moderate_balanced": {"shape": {"source_rows": 16, "assignments": 32, "k": 128, "n": 64, "experts": 4}, "outputs": 2048},
            "moderate_skewed": {"shape": {"source_rows": 16, "assignments": 32, "k": 128, "n": 64, "experts": 4}, "outputs": 2048},
            "weighted_balanced": {"shape": {"source_rows": 64, "assignments": 128, "k": 512, "n": 256, "experts": 8}, "outputs": 32768},
            "weighted_skewed": {"shape": {"source_rows": 64, "assignments": 128, "k": 512, "n": 256, "experts": 8}, "outputs": 32768},
        },
    },
    "kwv07-w4a16": {
        "suite_sha256": "782c84fd60b234d42e692f1c872c2590741fc2b6ee827d23336db5cc83eb1280",
        "contract_sha256": "a719306a1aad6f655385ee4c8f0418971674fa7ca717a17ca1dbb36ad2a1d208",
        "oracle_sha256": "4c6586eb1f4ab8b080153beeb267626037ca7fb62af89d5a2c73d12b80548afb",
        "cases": {
            "low_m": {"shape": {"m": 1, "k": 257, "n": 63, "group_size": 128}, "outputs": 63, "weight_family": "tail_adversarial", "reuse_cases": [1, 4, 16]},
            "moderate_m": {"shape": {"m": 16, "k": 257, "n": 63, "group_size": 128}, "outputs": 1008, "weight_family": "tail_adversarial", "reuse_cases": [1, 4]},
            "weighted_low_m": {"shape": {"m": 1, "k": 4096, "n": 1024, "group_size": 128}, "outputs": 1024, "weight_family": "shared_large_weights", "reuse_cases": [1, 4, 16]},
            "weighted_moderate_m": {"shape": {"m": 16, "k": 4096, "n": 1024, "group_size": 128}, "outputs": 16384, "weight_family": "shared_large_weights", "reuse_cases": [1, 4]},
        },
    },
}
KWV07_REUSE_ASSUMPTION = (
    "represented weights, scales, and activation remain resident and unchanged for each reuse count; "
    "cold setup includes one file read, allocation, and H2D transfer"
)
KWV07_TIMING_SCOPES = {
    "kwv07-grouped-gated": {
        "format": "KERNELWIKI-KWV07-TIMING-SCOPE/1",
        "family": "grouped",
        "device_event": "gather + both up/gate GEMMs for every nonempty expert + FP32 SiLU/product + scatter; stable assignment grouping is one-time host preparation outside the CUDA-event scope",
        "synchronized_host": "per-invocation submit through synchronized completion; stable assignment grouping is prepared once before sampled invocations",
        "setup_host": "input+oracle file read/parse, stable assignment grouping, allocations, H2D and sync",
        "components": {
            "group_metadata": "group_metadata_host_us records one-time stable assignment preparation measured by prep_us; it is inside one_time_setup_host_us and outside CUDA-event samples",
            "derived_one_use_host_total": "native cold_one_use_host_total_us is one_time_setup_host_us + median_host_us; a derived setup-plus-batch sum, not full cold request latency",
        },
    },
    "kwv07-w4a16": {
        "format": "KERNELWIKI-KWV07-TIMING-SCOPE/1",
        "family": "w4a16",
        "device_event": "first decode preparation and repeated compute are sampled separately as listed in components",
        "synchronized_host": "synchronized wall time for each listed operation scope",
        "setup_host": "input+oracle file read/parse, allocations, H2D and sync",
        "components": {
            "predecode": "predecode-w4 reports separate nine-sample medians in predecode_event_us and predecode_host_us for packed-nibble/scale decode to FP16; direct-w4 reports zero preparation",
            "reuse_compute": "dense cuBLAS after decode for predecode-w4; direct fused decode+SIMT dot for direct-w4",
            "derived_prep_plus_batch_device_total": "native cold_device_event_total_us is predecode_event_us median plus reuse-batch event median for predecode-w4, or reuse-batch event median for direct-w4; setup and input transfer are excluded",
            "derived_setup_prep_plus_batch_host_total": "native cold_host_wall_total_us is one_time_setup_host_us plus predecode_host_us median plus reuse-batch host median; this derived subtotal is not full cold invocation latency",
            "reuse_condition": "host-controlled resident-data reuse; unchanged activation/codes/scales within each case; no L2/HBM residency claim",
            "paired_M_control": "M=1 and M=16 within each K/N family share byte-identical packed codes/scales; activations are independent, not prefix-matched",
        },
    },
}
KWV07_ROLE_SPECS = {
    "kwv07-grouped-gated": {
        "suite": "grouped", "family": "grouped_gated", "outputs": 70144,
        "roles": {
            "grouped_correctness_baseline": ("baseline", "split-gated", "foreground-baseline", "check", None),
            "grouped_correctness_candidate": ("candidate", "paired-gated", "foreground-baseline", "check", None),
            "grouped_sanitizer_candidate": ("candidate", "paired-gated", "foreground-compute-sanitizer", "check", None),
            "grouped_baseline_timing": ("baseline", "split-gated", "foreground-baseline", "time", None),
            "grouped_candidate_timing": ("candidate", "paired-gated", "foreground-baseline", "time", None),
            "grouped_profile_candidate": ("candidate", "paired-gated", "foreground-ncu", "check", "weighted_skewed"),
        },
    },
    "kwv07-w4a16": {
        "suite": "w4a16", "family": "w4a16", "outputs": 18479,
        "roles": {
            "w4a16_correctness_baseline": ("baseline", "predecode-w4", "foreground-baseline", "check", None),
            "w4a16_correctness_candidate": ("candidate", "direct-w4", "foreground-baseline", "check", None),
            "w4a16_sanitizer_candidate": ("candidate", "direct-w4", "foreground-compute-sanitizer", "check", None),
            "w4a16_baseline_timing": ("baseline", "predecode-w4", "foreground-baseline", "time", None),
            "w4a16_candidate_timing": ("candidate", "direct-w4", "foreground-baseline", "time", None),
            "w4a16_profile_candidate": ("candidate", "direct-w4", "foreground-ncu", "check", "weighted_low_m"),
        },
    },
}


class EvidenceError(ValueError):
    pass


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value.lower()))


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


def _historical_source_root(raw: Any, current_root: Path, portable: bool) -> Path:
    """Resolve the recorded capture root without requiring it to exist locally."""
    if not portable:
        return current_root.resolve()
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise EvidenceError("manifest source_root must be an absolute historical path")
    return Path(raw).resolve()


def _capture_repo_relative(raw_path: str, root: Path, source_root: Path) -> Path | None:
    """Return a safe project-relative path, or None for a captured external identity."""
    raw = Path(raw_path)
    if raw.is_absolute():
        for capture_root in (source_root.resolve(), root.resolve()):
            try:
                relative = raw.relative_to(capture_root)
                break
            except ValueError:
                continue
        else:
            return None
    else:
        relative = raw
    if (not relative.parts or any(part in {"", ".", ".."} for part in relative.parts)):
        raise EvidenceError("controller project input escapes historical source_root")
    return relative


def _current_path_for_capture(raw_path: str, root: Path, source_root: Path) -> str:
    """Map paths inside either repository identity to this checkout; keep externals literal."""
    relative = _capture_repo_relative(raw_path, root, source_root)
    if relative is None:
        return str(Path(raw_path).resolve())
    mapped = (root / relative).resolve()
    if not mapped.is_relative_to(root.resolve()):
        raise EvidenceError("relative controller input escapes the relocated checkout")
    return str(mapped)


def _input_bindings(summary: dict[str, Any], root: Path | None = None,
                    source_root: Path | None = None) -> dict[str, str]:
    provenance = summary.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, dict) else None
    result: dict[str, str] = {}
    if isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str):
                if root is None or source_root is None:
                    key = str(Path(item["path"]).resolve())
                else:
                    key = _current_path_for_capture(item["path"], root, source_root)
                result[key] = item["sha256"]
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


def _captured_input_archives(base: Path, root: Path, case: dict[str, Any], required: bool
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
        if not archive_file.is_relative_to(base.resolve()):
            raise EvidenceError(f"{label}: archive path escapes the family evidence directory")
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


def _captured_project_file(root: Path, relative_path: str,
                           archives: dict[str, dict[str, Any]], label: str) -> Path:
    original = _repo_relative_identity(root, relative_path, f"{label} original path")
    archive = archives.get(str(original))
    if archive is not None:
        return archive["archive_path"]
    if not original.is_file():
        raise EvidenceError(f"{label}: project file is missing and has no captured archive")
    return original


def _controller_input_identity(path: Path,
                               archives: dict[str, dict[str, Any]]) -> str:
    resolved = path.resolve()
    for original, archive in archives.items():
        if Path(archive["archive_path"]).resolve() == resolved:
            return original
    return str(resolved)


def _exact_input_bindings(summary: dict[str, Any], expected: dict[str, str],
                          root: Path, source_root: Path,
                          allow_duplicate_paths: set[str] | None = None) -> bool:
    """Require archived capture aliases with their declared digests.

    Attention bundles may also appear as Project Control spec inputs. Those
    duplicates are accepted only when every occurrence carries the same digest.
    """
    provenance = summary.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, dict) else None
    if not isinstance(inputs, list):
        return False
    found: dict[str, list[str]] = {path: [] for path in expected}
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        try:
            resolved = _current_path_for_capture(item["path"], root, source_root)
        except EvidenceError:
            return False
        if resolved in found:
            found[resolved].append(item.get("sha256"))
    allowed = allow_duplicate_paths or set()
    return all(
        bool(found[path]) and all(value == sha for value in found[path])
        if path in allowed else found[path] == [sha]
        for path, sha in expected.items()
    )


def _validate_project_input_closure(summary: dict[str, Any], root: Path,
                                    source_root: Path, implementation: dict[str, Any],
                                    captured_archives: dict[str, dict[str, Any]],
                                    label: str,
                                    allow_duplicate_paths: set[str] | None = None) -> None:
    """Bind every repository-local controller input to the checkout or a declared archive."""
    provenance = summary.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, dict) else None
    if not isinstance(inputs, list):
        raise EvidenceError(f"{label}: controller input inventory is missing")
    executed_relative = implementation.get("executed_relative")
    binary_sha = implementation.get("binary_sha")
    if not isinstance(executed_relative, Path) or not _is_sha256(binary_sha):
        raise EvidenceError(f"{label}: archived executable identity is missing")
    seen: dict[str, str] = {}
    for item in inputs:
        if (not isinstance(item, dict) or not isinstance(item.get("path"), str) or
                not _is_sha256(item.get("sha256"))):
            raise EvidenceError(f"{label}: controller input path or SHA-256 is malformed")
        raw_path = item["path"]
        relative = _capture_repo_relative(raw_path, root, source_root)
        if relative is None:
            continue
        key = str((root / relative).resolve())
        input_sha = item["sha256"]
        if key in seen:
            if key not in (allow_duplicate_paths or set()) or seen[key] != input_sha:
                raise EvidenceError(f"{label}: duplicate repository input identity {relative.as_posix()}")
            continue
        seen[key] = input_sha
        if relative == executed_relative:
            if input_sha != binary_sha or digest(implementation["binary_path"]) != binary_sha:
                raise EvidenceError(f"{label}: executed binary input differs from its archived ELF")
            continue
        archive = captured_archives.get(key)
        if archive is not None:
            if input_sha != archive["sha256"]:
                raise EvidenceError(f"{label}: repository input differs from its captured archive")
            continue
        current_file = (root / relative).resolve()
        if (not current_file.is_relative_to(root.resolve()) or not current_file.is_file() or
                digest(current_file) != input_sha):
            raise EvidenceError(f"{label}: repository input is absent or mismatched and has no archive: {relative.as_posix()}")


def _binary_binding(summary: dict[str, Any], root: Path, source_root: Path,
                    portable: bool, expected_path: Path, expected_sha: str) -> None:
    binaries = summary.get("binaries")
    if not isinstance(binaries, list):
        raise EvidenceError("controller receipt does not list binary_paths")
    def captured_path(raw: Any) -> Path | None:
        if not isinstance(raw, str) or not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = (source_root if portable else root) / path
        return path.resolve()

    target = expected_path.resolve()
    if not any(isinstance(item, dict) and captured_path(item.get("path")) == target and
               item.get("sha256") == expected_sha for item in binaries):
        raise EvidenceError(f"controller receipt is not bound to binary {expected_path}")
    provenance = summary.get("provenance")
    binary = provenance.get("binary") if isinstance(provenance, dict) else None
    if (not isinstance(binary, dict) or binary.get("sha256") != expected_sha or
            captured_path(binary.get("path")) != target):
        raise EvidenceError("controller argv binary hash differs from the declared implementation")


def _controller_run(base: Path, root: Path, source_root: Path, portable: bool,
                    run: Any, label: str, expected_kind: str,
                    implementation: dict[str, Any], contract: Path, contract_sha: str,
                    inputs_path: Path, inputs_sha: str, oracle_path: Path,
                    oracle_sha: str, gpu_uuid: str,
                    evidence_ids: set[str], captured_inputs: dict[str, str] | None = None,
                    captured_archives: dict[str, dict[str, Any]] | None = None,
                    additional_inputs: dict[str, str] | None = None,
                    allow_duplicate_captured_inputs: set[str] | None = None
                    ) -> dict[str, Any]:
    if not isinstance(run, dict):
        raise EvidenceError(f"{label}: controller receipt is missing")
    execution_path = checked_file(base, run.get("execution"), f"{label} execution")
    evidence_path = checked_file(base, run.get("evidence"), f"{label} evidence")
    lease_record = run.get("lease")
    if not isinstance(lease_record, dict):
        raise EvidenceError(f"{label}: lease receipt is missing")
    if portable:
        lease_path = checked_file(base, lease_record, f"{label} archived lease")
        original_lease_path = lease_record.get("original_path")
        if not isinstance(original_lease_path, str) or not Path(original_lease_path).is_absolute():
            raise EvidenceError(f"{label}: archived lease lacks its absolute original_path")
        original_lease_path = str(Path(original_lease_path).resolve())
    else:
        # Legacy manifests point directly into the local Project Control runtime.
        lease_path = (root / str(lease_record.get("path", ""))).resolve()
        if not lease_path.is_relative_to(root) or not lease_path.is_file():
            raise EvidenceError(f"{label}: controller lease receipt is unavailable")
        if digest(lease_path) != lease_record.get("sha256"):
            raise EvidenceError(f"{label}: lease SHA-256 mismatch")
        original_lease_path = str(lease_path)

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
            str(Path(execution_lease).resolve()) != original_lease_path):
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
    if (not isinstance(source, dict) or
            Path(str(source.get("project_root", ""))).resolve() != source_root):
        raise EvidenceError(f"{label}: controller source snapshot is unbound to the declared source_root")
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
    actual_inputs = _input_bindings(summary, root, source_root)
    if portable:
        _validate_project_input_closure(summary, root, source_root, implementation,
                                        captured_archives or {}, label,
                                        allow_duplicate_captured_inputs)
    if any(actual_inputs.get(path) != value for path, value in expected_inputs.items()):
        raise EvidenceError(f"{label}: controller inputs do not bind the contract, represented data, and source")
    if captured_inputs and not _exact_input_bindings(
            summary, captured_inputs, root, source_root, allow_duplicate_captured_inputs):
        raise EvidenceError(f"{label}: controller inputs do not exactly bind archived source aliases")
    _binary_binding(summary, root, source_root, portable,
                    implementation["executed_path"], implementation["binary_sha"])

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
            Path(str(lease.get("project_root", ""))).resolve() != source_root or
            expected_resource not in lease.get("resource_ids", [])):
        raise EvidenceError(f"{label}: foreground lease is not bound to the declared GPU and source_root")
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


def _is_kwv07_contract(contract: dict[str, Any]) -> bool:
    return contract.get("format") == KWV07_SUITE_CONTRACT_FORMAT


def _is_kwv08_contract(contract: dict[str, Any]) -> bool:
    return contract.get("format") == KWV08_CONTRACT_FORMAT


def _kwv08_finite_nonnegative(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(float(value)) and float(value) >= 0.0)


def _kwv08_native_checks(result: dict[str, Any], records: dict[str, Any],
                          gpu_uuid: str, role: str, label: str) -> None:
    """Check recurrence app JSON against the frozen bundle and role invocation."""
    spec = KWV08_ROLES.get(role)
    if spec is None:
        raise EvidenceError(f"{label}: unknown KWV08 role")
    _kind, implementation, variant, mode, selected_case = spec
    expected_uuid = gpu_uuid.removeprefix("GPU-").replace("-", "").lower()
    if (not isinstance(result, dict) or
            result.get("format") != KWV08_NATIVE_FORMAT or
            result.get("status") != "passed" or
            result.get("variant") != variant or result.get("mode") != mode or
            str(result.get("compute_capability")) != "7.0" or
            result.get("gpu_uuid") != expected_uuid or
            not isinstance(result.get("gpu"), str) or not result["gpu"] or
            result.get("bundle_file_sha256") != records["bundle_file_sha256"] or
            result.get("bundle_payload_sha256") != records["bundle_payload_sha256"]):
        raise EvidenceError(f"{label}: native runner identity, status, GPU, or bundle binding failed")
    tolerance = result.get("tolerance")
    if (not isinstance(tolerance, dict) or
            tolerance.get("atol") != records["atol"] or
            tolerance.get("rtol") != records["rtol"]):
        raise EvidenceError(f"{label}: native numerical tolerance differs from the frozen contract")
    expected_semantics = {
        "host_enqueue_us": "wall time through submission of all launches",
        "host_completion_us": "wall time through stream completion",
        "device_event_interval_us": "CUDA event interval; can include device idle while host submits per-token launches",
    }
    if result.get("timing_semantics") != expected_semantics:
        raise EvidenceError(f"{label}: native timing scope differs from the frozen whole-operator definitions")

    expected_ids = [selected_case] if selected_case else list(records["cases"])
    cases = result.get("cases")
    if (not isinstance(cases, list) or
            [item.get("case_id") if isinstance(item, dict) else None for item in cases] != expected_ids):
        raise EvidenceError(f"{label}: native case set or order differs from the role suite")
    for item in cases:
        case_id = item["case_id"]
        expected = records["cases"].get(case_id)
        if not isinstance(expected, dict):
            raise EvidenceError(f"{label}: native runner returned an undeclared case")
        expected_shape = {key: expected[key] for key in ("T", "B", "d_v", "d_k")}
        shape = item.get("shape")
        if (not isinstance(shape, dict) or
                any(type(shape.get(key)) is not int for key in expected_shape) or
                shape != expected_shape or item.get("status") != "passed" or
                item.get("variant") != variant or item.get("error") not in (None, "")):
            raise EvidenceError(f"{label} {case_id}: native case status or shape differs from contract")
        checks = item.get("checks")
        expected_checks = list(KWV08_CHECKS)
        if mode == "profile":
            expected_checks = expected_checks[:3]
        if (not isinstance(checks, list) or
                [check.get("name") if isinstance(check, dict) else None for check in checks] != expected_checks):
            raise EvidenceError(f"{label} {case_id}: native full-output/final-state check set is incomplete")
        expected_counts = {
            "outputs": expected["live_steps"] * expected["d_v"],
            "padding": (expected["T"] * expected["B"] - expected["live_steps"]) * expected["d_v"],
            "state": expected["B"] * expected["d_v"] * expected["d_k"],
        }
        for check in checks:
            kind = KWV08_CHECKS[check["name"]]
            if (any(type(check.get(field)) is not int for field in ("checked", "failed", "nonfinite")) or
                    check.get("status") != "passed" or
                    check.get("checked") != expected_counts[kind] or
                    check.get("failed") != 0 or check.get("nonfinite") != 0 or
                    not _kwv08_finite_nonnegative(check.get("max_abs_error")) or
                    not _kwv08_finite_nonnegative(check.get("max_scaled_error"))):
                raise EvidenceError(f"{label} {case_id}: native {check['name']} did not check its full expected extent")
            if kind in {"outputs", "state"} and check["max_scaled_error"] > 1.0:
                raise EvidenceError(f"{label} {case_id}: native {check['name']} exceeds frozen tolerance")
            if kind == "padding" and (check["max_abs_error"] != 0 or check["max_scaled_error"] != 0):
                raise EvidenceError(f"{label} {case_id}: padded output sentinel changed")

        timing = item.get("timing")
        if not isinstance(timing, dict):
            raise EvidenceError(f"{label} {case_id}: native timing record is missing")
        if mode == "time":
            fields = ("host_enqueue_us", "host_completion_us", "device_event_interval_us")
            if timing.get("status") != "passed":
                raise EvidenceError(f"{label} {case_id}: native timing did not pass")
            for field, median_field in zip(fields, (
                    "host_enqueue_median_us", "host_completion_median_us",
                    "device_event_interval_median_us")):
                samples = timing.get(field)
                if (not isinstance(samples, list) or len(samples) != 7 or
                        any(not _kwv08_finite_nonnegative(value) or float(value) <= 0 for value in samples)):
                    raise EvidenceError(f"{label} {case_id}: native {field} lacks seven positive samples")
                median = timing.get(median_field)
                expected_median = statistics.median(float(value) for value in samples)
                if (not _kwv08_finite_nonnegative(median) or
                        not math.isclose(float(median), expected_median, rel_tol=1e-9, abs_tol=1e-9)):
                    raise EvidenceError(f"{label} {case_id}: native {median_field} differs from its samples")
        elif timing.get("status") != "not_run":
            raise EvidenceError(f"{label} {case_id}: correctness/profile run unexpectedly contains timing evidence")


def _kwv08_suite_records(base: Path, root: Path, case: dict[str, Any],
                         contract_path: Path, contract: dict[str, Any],
                         input_path: Path, input_sha: str,
                         oracle_path: Path, oracle_sha: str) -> dict[str, Any]:
    """Bind the recurrence contract, CPU oracle, and represented bundle to fixed shapes."""
    case_id = str(case.get("id", "recurrence"))
    expected_signature = "b4aae5c2c268dc0099b08cb4c375010ba1e9269725de11c440d80bf33bfcabcd"
    expected_cases = [
        {"case_id": "decode_many_sequences", "T": 1, "B": 256, "d_v": 32, "d_k": 32, "live_steps": 137},
        {"case_id": "short_prefill_many_sequences", "T": 33, "B": 128, "d_v": 32, "d_k": 32, "live_steps": 1845},
        {"case_id": "long_prefill_many_sequences", "T": 257, "B": 128, "d_v": 64, "d_k": 64, "live_steps": 13014},
    ]
    if (case_id != "recurrence_state_ownership_bundle" or
            contract.get("format") != KWV08_CONTRACT_FORMAT or
            contract.get("case_signature") != expected_signature or
            contract.get("subcases") != expected_cases or
            contract.get("numerical_contract") != {
                "accumulation": "FP32 native; scalar FP64 independent semantic oracle",
                "active_key_norm": "unit norm in represented FP32 bundle; reader verifies; native kernel does not normalize",
                "atol": 2e-05,
                "comparison": "abs(actual-reference) <= atol + rtol*abs(reference) for every live output and every final-state element",
                "exceptional_values": "active inputs and initial states finite; padding NaNs are excluded from live operation",
                "gate_range": "0 <= alpha,beta <= 1, includes exact zero and near-one values",
                "layout": "time-major interleaved [T,B,d_k/d_v], state row-major [B,d_v,d_k]",
                "operator": "D=alpha*S_previous; e=v-D@k; S=D+beta*outer(e,k); o=S@q",
                "output_state": "post-update",
                "padding": "NaN input/reference sentinels only outside each sequence length; output sentinel must remain untouched",
                "represented_inputs": "FP32 keys, values, queries, alpha, beta, and initial state; FP64 independent recurrence outputs/final states serialized in the same bundle",
                "rtol": 0.0002,
                "sequence_contract": "all live timesteps in order, preserve nonzero initial state, carry final state across segments, no history truncation or sequence mixing",
                "state_orientation": "S[d_v,d_k]",
            }):
        raise EvidenceError(f"{case_id}: contract identity, subcases, or numerical policy differs from frozen KWV08 contract")
    represented = contract.get("represented_input")
    if (not isinstance(represented, dict) or
            represented.get("path") != "benchmark_runs/volta-recurrence/inputs/recurrence_inputs.bin" or
            input_sha != KWV08_INPUT_SHA256 or represented.get("sha256") != input_sha or
            digest(input_path) != input_sha or input_path.stat().st_size != 56_285_400):
        raise EvidenceError(f"{case_id}: recurrence represented-input identity or size differs from frozen bundle")

    bundle_manifest_path = base / "recurrence_inputs.json"
    bundle_manifest = read_json(bundle_manifest_path)
    manifest_cases = bundle_manifest.get("cases") if isinstance(bundle_manifest, dict) else None
    if (not isinstance(bundle_manifest, dict) or
            bundle_manifest.get("format") != "KERNELWIKI-VOLTA-RECURRENCE-BUNDLE/1" or
            bundle_manifest.get("bundle_file_sha256") != input_sha or
            bundle_manifest.get("bundle_payload_sha256") != KWV08_PAYLOAD_SHA256 or
            bundle_manifest.get("bundle_bytes") != input_path.stat().st_size or
            not isinstance(manifest_cases, list) or len(manifest_cases) != len(expected_cases)):
        raise EvidenceError(f"{case_id}: bundle manifest identity differs from the frozen represented data")
    for actual, expected in zip(manifest_cases, expected_cases):
        if (not isinstance(actual, dict) or
                any(actual.get(key) != expected[key] for key in ("case_id", "T", "B", "d_v", "d_k", "live_steps")) or
                actual.get("length_min") != 0 or actual.get("length_max") != expected["T"]):
            raise EvidenceError(f"{case_id}: bundle manifest case dimensions or ragged lengths differ")

    oracle = read_json(oracle_path)
    if (oracle.get("format") != "KERNELWIKI-CPU-ORACLE/1" or
            oracle.get("case_id") != case_id or oracle.get("passed") is not True or
            oracle.get("represented_input") != represented):
        raise EvidenceError(f"{case_id}: CPU oracle is not bound to the same passing represented bundle")
    checks = oracle.get("checks")
    required_oracle_checks = {
        "expanded_rank_one_equation_small_semantic_case",
        "segmented_carry_all_bundle_outputs",
        "segmented_carry_all_bundle_final_states",
    }
    if (not isinstance(checks, list) or
            {item.get("name") for item in checks if isinstance(item, dict)} != required_oracle_checks):
        raise EvidenceError(f"{case_id}: independent CPU oracle lacks equation and segmented output/state checks")
    for item in checks:
        if not isinstance(item, dict):
            raise EvidenceError(f"{case_id}: CPU oracle contains a malformed comparison")
        error, reference, atol, rtol = (item.get(key) for key in
                                        ("max_abs_error", "reference_max_abs", "atol", "rtol"))
        if (not all(_kwv08_finite_nonnegative(value) for value in (error, reference, atol, rtol)) or
                float(error) > float(atol) + float(rtol) * float(reference)):
            raise EvidenceError(f"{case_id}: independent CPU oracle comparison fails its stated tolerance")
    probe = read_json(base / "fp32_policy_probe.json")
    if (probe.get("format") != "KERNELWIKI-RECURRENCE-FP32-POLICY-PROBE/1" or
            probe.get("gpu_execution") is not False or
            probe.get("bundle_file_sha256") != input_sha or
            not isinstance(probe.get("cases"), list) or len(probe["cases"]) != len(expected_cases) or
            any(not isinstance(item, dict) or item.get("within_predeclared_policy") is not True
                for item in probe["cases"])):
        raise EvidenceError(f"{case_id}: CPU-only FP32 policy probe is incomplete or failed")
    case_matrix = read_json(base / "case_matrix.json")
    matrix_cases = case_matrix.get("bundle", {}).get("cases") if isinstance(case_matrix, dict) else None
    if matrix_cases != expected_cases:
        raise EvidenceError(f"{case_id}: case matrix differs from the frozen recurrence bundle")
    return {
        "cases": {row["case_id"]: row for row in expected_cases},
        "bundle_file_sha256": input_sha,
        "bundle_payload_sha256": KWV08_PAYLOAD_SHA256,
        "atol": 2e-05,
        "rtol": 2e-4,
        "oracle_path": oracle_path,
        "oracle_sha256": oracle_sha,
        "bundle_path": input_path,
    }


def _kwv08_source_closure(root: Path, case: dict[str, Any], base: Path | None = None,
                          source_root: Path | None = None, portable: bool = False
                          ) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    source_root = source_root or root
    case_id = str(case.get("id", "recurrence"))
    closure_path, closure_sha = sha_binding(root, case.get("source_closure"),
                                            f"{case_id} source closure")
    closure = read_json(closure_path)
    repository_inputs = closure.get("repository_inputs") if isinstance(closure, dict) else None
    toolchain_inputs = closure.get("toolchain_inputs") if isinstance(closure, dict) else None
    declarations = closure.get("declaring_specs") if isinstance(closure, dict) else None
    if (closure_sha != KWV08_SOURCE_CLOSURE_SHA256 or
            not isinstance(closure, dict) or closure.get("format") != KWV08_SOURCE_CLOSURE_FORMAT or
            not isinstance(repository_inputs, list) or not repository_inputs or
            not isinstance(toolchain_inputs, list) or not toolchain_inputs or
            not isinstance(declarations, list) or len(declarations) != len(KWV08_ROLES)):
        raise EvidenceError(f"{case_id}: source closure format or six-role declaration set is invalid")

    captured = _captured_input_archives(base or root, root, case, required=True)
    expected: dict[str, tuple[str, str, int]] = {}
    for index, item in enumerate(repository_inputs):
        label = f"{case_id} source closure repository input {index}"
        if not isinstance(item, dict):
            raise EvidenceError(f"{label}: malformed repository input")
        original = _repo_relative_identity(root, item.get("original_path"), f"{label} original path")
        archive_raw, expected_sha, size = item.get("archived_path"), item.get("sha256"), item.get("size_bytes")
        if (not isinstance(archive_raw, str) or not isinstance(expected_sha, str) or len(expected_sha) != 64 or
                type(size) is not int or size < 0):
            raise EvidenceError(f"{label}: archive identity is malformed")
        archive_path = absolute_repo_file(root, archive_raw, f"{label} archive")
        if (archive_path.stat().st_size != size or digest(archive_path) != expected_sha or
                str(original) in expected):
            raise EvidenceError(f"{label}: archived hash, size, or original identity differs")
        expected[str(original)] = (str(archive_path), expected_sha, size)
    if set(expected) != set(captured):
        raise EvidenceError(f"{case_id}: captured input aliases differ from repository source closure")
    for original, (archive, expected_sha, size) in expected.items():
        capture = captured[original]
        if (str(capture["archive_path"]) != archive or capture["sha256"] != expected_sha or
                capture["archive_path"].stat().st_size != size):
            raise EvidenceError(f"{case_id}: captured alias differs from source closure")

    expected_toolchain: dict[str, str] = {}
    for index, item in enumerate(toolchain_inputs):
        label = f"{case_id} source closure toolchain input {index}"
        if not isinstance(item, dict):
            raise EvidenceError(f"{label}: malformed external toolchain input")
        raw_path, expected_sha, size = item.get("path"), item.get("sha256"), item.get("size_bytes")
        if (not isinstance(raw_path, str) or not Path(raw_path).is_absolute() or
                not isinstance(expected_sha, str) or len(expected_sha) != 64 or
                type(size) is not int or size < 0 or raw_path in expected_toolchain):
            raise EvidenceError(f"{label}: toolchain identity is malformed or duplicated")
        path = Path(raw_path)
        if (not portable and
                (not path.is_file() or path.stat().st_size != size or digest(path) != expected_sha)):
            raise EvidenceError(f"{label}: current toolchain file differs from captured identity")
        expected_toolchain[raw_path] = expected_sha

    expected_roles = {f"recurrence_{role}" for role in KWV08_ROLES}
    expected_specs = {
        f"benchmark_runs/volta-recurrence/controller/role-{role}.spec.json"
        for role in expected_roles
    }
    declared: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(declarations):
        label = f"{case_id} source closure role spec {index}"
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise EvidenceError(f"{label}: malformed spec identity")
        original = _repo_relative_identity(root, item["path"], f"{label} path")
        capture = captured.get(str(original))
        if (capture is None or capture["sha256"] != item.get("sha256") or
                original.relative_to(root).as_posix() not in expected_specs):
            raise EvidenceError(f"{label}: spec is not an exact captured KWV08 role spec")
        spec = read_json(capture["archive_path"])
        role = spec.get("role") if isinstance(spec, dict) else None
        if role not in expected_roles or role in {entry.get("role") for entry in declared.values()}:
            raise EvidenceError(f"{label}: spec role is unknown or duplicated")
        inputs = spec.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise EvidenceError(f"{label}: archived role spec inputs are missing")
        for raw_input in inputs:
            if not isinstance(raw_input, str) or not raw_input:
                raise EvidenceError(f"{label}: declared input path is malformed")
            if Path(raw_input).is_absolute():
                if raw_input not in expected_toolchain:
                    raise EvidenceError(f"{label}: absolute input is absent from toolchain closure")
            else:
                input_identity = _repo_relative_identity(root, raw_input, f"{label} declared input")
                if str(input_identity) not in captured:
                    raise EvidenceError(f"{label}: repository input is absent from source closure")
        declared[str(original)] = {"role": role, "path": original, "spec": spec}
    if ({str(record["path"].relative_to(root)) for record in declared.values()} != expected_specs or
            {record["role"] for record in declared.values()} != expected_roles):
        raise EvidenceError(f"{case_id}: source closure does not declare each exact role spec")
    input_sets = [set(record["spec"]["inputs"]) for record in declared.values()]
    if any(values != input_sets[0] for values in input_sets[1:]):
        raise EvidenceError(f"{case_id}: six role specs do not share one immutable input set")
    repository_spec_inputs = {
        str(_repo_relative_identity(root, value, f"{case_id} role input identity"))
        for value in input_sets[0] if not Path(value).is_absolute()
    }
    external_spec_inputs = {value for value in input_sets[0] if Path(value).is_absolute()}
    if (repository_spec_inputs != set(captured) or external_spec_inputs != set(expected_toolchain)):
        raise EvidenceError(f"{case_id}: source closure does not cover every declared role input")

    gpu_uuid = str(case.get("gpu_uuid", ""))
    if (closure.get("project_root") != str(source_root) or
            closure.get("gpu_uuid_target") != gpu_uuid):
        raise EvidenceError(f"{case_id}: source closure root or target GPU differs from the summary")
    return ({"path": closure_path, "sha256": closure_sha, "data": closure,
             "declared_specs": {record["role"]: record for record in declared.values()}},
            captured, expected_toolchain)


def _kwv08_validate_spec(root: Path, role: str, spec_record: dict[str, Any],
                         binary_path: Path, gpu_uuid: str,
                         source_root: Path | None = None) -> None:
    source_root = source_root or root
    spec = spec_record["spec"]
    _kind, _implementation_name, variant, mode, selected_case = KWV08_ROLES[role]
    native_role = f"recurrence_{role}"
    recipe = "ncu" if _kind == "foreground-ncu" else (
        "compute-sanitizer" if _kind == "foreground-compute-sanitizer" else "baseline")
    expected_argv = [str(binary_path.relative_to(source_root)), "--bundle",
                      "benchmark_runs/volta-recurrence/inputs/recurrence_inputs.bin",
                      "--variant", variant, "--mode", mode]
    if selected_case is not None:
        expected_argv.extend(["--case", selected_case])
    if (spec.get("role") != native_role or
            spec.get("contract_id") != "kwv08-recurrence-state-ownership" or
            spec.get("recipe") != recipe or spec.get("argv") != expected_argv or
        spec.get("project_root") != str(source_root) or spec.get("command_cwd") != str(source_root) or
            spec.get("resources", {}).get("gpu_uuids") != [gpu_uuid]):
        raise EvidenceError(f"recurrence {role}: captured controller spec differs from the fixed native invocation")
    benchmark = spec.get("benchmark")
    build_argv = benchmark.get("build_argv") if isinstance(benchmark, dict) else None
    if role == "correctness_baseline":
        script = str((source_root / "benchmark_runs/volta-recurrence/source/capture_nvcc_build.sh").resolve())
        source = str((source_root / "artifacts/kernels/volta-recurrence/variants/gdn_runner.cu").resolve())
        if (not isinstance(build_argv, list) or len(build_argv) < 8 or
                build_argv[0] != script or "--" not in build_argv or
                "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc" not in build_argv or
                "-arch=sm_70" not in build_argv or source not in build_argv or
                build_argv[-2:] != ["-o", str(binary_path.resolve())]):
            raise EvidenceError("recurrence baseline correctness role does not own the one SM70 build")
    elif build_argv:
        raise EvidenceError(f"recurrence {role}: shared executable was rebuilt after baseline correctness")


def _kwv08_validate_argv(summary: dict[str, Any], root: Path, role: str,
                         implementation: dict[str, Any],
                         source_root: Path | None = None) -> None:
    source_root = source_root or root
    _kind, _implementation_name, variant, mode, selected_case = KWV08_ROLES[role]
    provenance = summary.get("provenance")
    argv = provenance.get("argv") if isinstance(provenance, dict) else None
    if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
        raise EvidenceError(f"recurrence {role}: controller application argv is missing")
    executable = Path(argv[0])
    if not executable.is_absolute():
        executable = source_root / executable
    if executable.resolve() != implementation["executed_path"].resolve():
        raise EvidenceError(f"recurrence {role}: executed binary differs from the shared measured ELF")
    expected = ["--bundle", "benchmark_runs/volta-recurrence/inputs/recurrence_inputs.bin",
                "--variant", variant, "--mode", mode]
    if selected_case is not None:
        expected += ["--case", selected_case]
    if argv[1:] != expected:
        raise EvidenceError(f"recurrence {role}: native argv does not match its fixed bundle/variant/mode selection")


def _kwv08_archive_hash(manifest: dict[str, Any], path: Path, role_dir: Path,
                        label: str) -> str:
    try:
        relative = path.resolve().relative_to(role_dir.resolve()).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"{label}: archived file is outside its role directory") from exc
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise EvidenceError(f"{label}: derived archive file inventory is missing")
    expected = files.get(relative)
    actual = digest(path)
    if not isinstance(expected, str) or expected != actual:
        raise EvidenceError(f"{label}: archive index does not bind {relative}")
    return actual


def _kwv08_copied_role_file(root: Path, role_dir: Path, raw_path: str,
                            label: str, source_root: Path | None = None,
                            portable: bool = False) -> Path:
    """Resolve a collector file path while keeping it inside the archived role."""
    supplied = Path(raw_path)
    role_root = role_dir.resolve()
    if supplied.is_absolute():
        candidate = supplied.resolve()
        if portable:
            source_root = source_root or root
            historical_role = (source_root / role_dir.resolve().relative_to(root.resolve())).resolve()
            try:
                candidate = (role_root / candidate.relative_to(historical_role)).resolve()
            except ValueError:
                pass
    else:
        repo_candidate = (root / supplied).resolve()
        try:
            repo_candidate.relative_to(role_root)
            candidate = repo_candidate
        except ValueError:
            candidate = (role_dir / supplied).resolve()
    try:
        candidate.relative_to(role_root)
    except ValueError as exc:
        raise EvidenceError(f"{label}: captured file escapes its role archive") from exc
    if not candidate.is_file():
        raise EvidenceError(f"{label}: captured file is missing")
    return candidate


def _kwv08_bind_stdout_capture(base: Path, root: Path, role: str, gpu_uuid: str,
                               run: dict[str, Any], native_stdout: Path,
                               source_root: Path | None = None,
                               portable: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = source_root or root
    archive_path = checked_file(base, run.get("archive_manifest"), f"recurrence {role} archive manifest")
    archive = read_json(archive_path)
    role_dir = archive_path.parent
    role_name = f"recurrence_{role}"
    files = archive.get("files") if isinstance(archive, dict) else None
    if (not isinstance(archive, dict) or archive.get("format") != "KERNELWIKI-KWV08-ARCHIVE/1" or
            archive.get("role") != role_name or not isinstance(files, dict)):
        raise EvidenceError(f"recurrence {role}: derived archive manifest identity is malformed")
    actual_files: dict[str, str] = {}
    for path in sorted(role_dir.rglob("*")):
        if path.is_symlink():
            raise EvidenceError(f"recurrence {role}: role archive contains a symlink")
        if path.is_file() and path.resolve() != archive_path.resolve():
            actual_files[path.relative_to(role_dir).as_posix()] = digest(path)
    if files != actual_files:
        raise EvidenceError(f"recurrence {role}: archive index does not exactly inventory captured role files")

    role_manifest_path = role_dir / "role-manifest.json"
    role_manifest = read_json(role_manifest_path)
    _kwv08_archive_hash(archive, role_manifest_path, role_dir, f"recurrence {role} role manifest")
    if (not isinstance(role_manifest, dict) or
            role_manifest.get("format") != "KERNELWIKI-KWV08-NATIVE-ROLE/1" or
            role_manifest.get("role") != role_name or
            role_manifest.get("evidence_id") != archive.get("evidence_id") or
            role_manifest.get("controller_status") != "succeeded" or
            role_manifest.get("controller_returncode") != 0 or
            role_manifest.get("controller_result_status") != "succeeded" or
            archive.get("runner_status") != "passed" or
            not isinstance(role_manifest.get("copied_files"), list)):
        raise EvidenceError(f"recurrence {role}: controller role manifest identity differs")
    role_binary = role_manifest.get("binary")
    archive_binary = archive.get("binary")
    if (not isinstance(role_binary, dict) or not isinstance(archive_binary, dict) or
            any(role_binary.get(key) != archive_binary.get(key) for key in ("path", "sha256", "size_bytes")) or
            role_binary.get("available") is not True):
        raise EvidenceError(f"recurrence {role}: role and derived archive manifests disagree on shared ELF")
    archived_binary = absolute_repo_file(root, archive_binary.get("path"), f"recurrence {role} archived ELF")
    if (digest(archived_binary) != archive_binary.get("sha256") or
            archived_binary.stat().st_size != archive_binary.get("size_bytes")):
        raise EvidenceError(f"recurrence {role}: archived ELF bytes differ from role manifest")
    for item in role_manifest.get("copied_files", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise EvidenceError(f"recurrence {role}: copied-file receipt is malformed")
        archived = _kwv08_copied_role_file(root, role_dir, item["path"],
                                           f"recurrence {role} copied file",
                                           source_root=source_root, portable=portable)
        archived = checked_file(base, {"path": archived.relative_to(base).as_posix(),
                                       "sha256": item.get("sha256")},
                                f"recurrence {role} copied file")
        _kwv08_archive_hash(archive, archived, role_dir, f"recurrence {role} copied file")
        if type(item.get("size_bytes")) is not int or archived.stat().st_size != item["size_bytes"]:
            raise EvidenceError(f"recurrence {role}: copied-file size differs from role manifest")

    evidence_path = checked_file(base, run.get("evidence"), f"recurrence {role} controller export")
    exported = read_json(evidence_path)
    evidence = exported.get("evidence") if isinstance(exported, dict) else None
    artifacts = evidence.get("artifacts") if isinstance(evidence, dict) else None
    evidence_id = evidence.get("id") if isinstance(evidence, dict) else None
    controller_path = checked_file(base, run.get("execution"), f"recurrence {role} controller JSON")
    controller = read_json(controller_path)
    lease_record = run.get("lease")
    if not isinstance(lease_record, dict):
        raise EvidenceError(f"recurrence {role}: lease binding is missing")
    lease_path = (checked_file(base, lease_record, f"recurrence {role} archived lease") if portable
                  else absolute_repo_file(root, lease_record.get("path"), f"recurrence {role} lease"))
    if (archive.get("evidence_id") != evidence_id or controller.get("evidence_id") != evidence_id or
            digest(controller_path) != files.get("controller-run.json") or
            digest(evidence_path) != files.get("controller-evidence.json") or
            digest(lease_path) != files.get("lease.json") or digest(lease_path) != lease_record.get("sha256")):
        raise EvidenceError(f"recurrence {role}: controller export, execution, or lease differs from archive index")
    raw_export = checked_file(base, run.get("project_control_raw"), f"recurrence {role} raw Project Control export")
    if digest(raw_export) != files.get("project-control-read-export.json"):
        raise EvidenceError(f"recurrence {role}: raw Project Control export differs from archive index")

    summary = evidence.get("summary") if isinstance(evidence, dict) else None
    record = summary.get("record") if isinstance(summary, dict) else None
    stdout_path = record.get("stdout") if isinstance(record, dict) else None
    stdout_artifacts = [item for item in artifacts if isinstance(item, dict) and item.get("kind") == "stdout"] if isinstance(artifacts, list) else []
    outer = role_dir / "foreground.stdout.txt"
    outer_hash = files.get("foreground.stdout.txt")
    expected_uuid = gpu_uuid.removeprefix("GPU-").replace("-", "").lower()
    if (archive.get("gpu_uuid_raw_hex") != expected_uuid or
            str(archive.get("compute_capability")) != "7.0" or
            not isinstance(stdout_path, str) or len(stdout_artifacts) != 1 or
            stdout_artifacts[0].get("complete") != 1 or
            stdout_artifacts[0].get("path") != stdout_path or
            stdout_artifacts[0].get("content_hash") != outer_hash or
            not isinstance(outer_hash, str) or digest(outer) != outer_hash):
        raise EvidenceError(f"recurrence {role}: controller stdout is not hash-bound to its captured artifact")
    copied_stdout = [item for item in role_manifest.get("copied_files", [])
                     if isinstance(item, dict) and
                     Path(str(item.get("path", ""))).name == "foreground.stdout.txt"]
    if (len(copied_stdout) != 1 or copied_stdout[0].get("source_path") != stdout_path or
            copied_stdout[0].get("sha256") != outer_hash):
        raise EvidenceError(f"recurrence {role}: archived outer stdout path differs from controller artifact identity")

    capture = run.get("native_stdout_capture")
    capture_path = checked_file(base, capture, f"recurrence {role} native stdout capture")
    if capture_path.resolve() != native_stdout.resolve():
        raise EvidenceError(f"recurrence {role}: app stdout path differs from declared native capture")
    capture_hash = _kwv08_archive_hash(archive, capture_path, role_dir,
                                       f"recurrence {role} native stdout")
    if capture.get("sha256") != capture_hash:
        raise EvidenceError(f"recurrence {role}: native stdout capture hash differs")
    if role in {"correctness_baseline", "correctness_candidate", "baseline_timing", "candidate_timing"}:
        if (capture_path.relative_to(role_dir).as_posix() != "foreground.stdout.txt" or
                capture.get("reported_path") is not None or capture_hash != outer_hash or
                stdout_artifacts[0].get("content_hash") != capture_hash):
            raise EvidenceError(f"recurrence {role}: native stdout is not the captured controller stdout bytes")
    else:
        try:
            relative_stdout = capture_path.relative_to(role_dir).as_posix()
            outer_text = outer.read_text(encoding="utf-8", errors="replace")
        except (ValueError, OSError) as exc:
            raise EvidenceError(f"recurrence {role}: nested native stdout is outside the role archive") from exc
        reported = capture.get("reported_path")
        expected_relative = ("sanitizer/run/stdout.txt" if role == "sanitizer_candidate"
                             else "ncu/run/command.stdout.txt")
        marker = "stdout: " if role == "sanitizer_candidate" else "command stdout: "
        if (relative_stdout != expected_relative or not isinstance(reported, str) or
                not Path(reported).is_absolute() or f"{marker}{reported}" not in outer_text):
            raise EvidenceError(f"recurrence {role}: nested native stdout path is not identified by controller output")
        if role == "sanitizer_candidate":
            path_report_path = role_dir / "sanitizer-path-report.json"
            path_report = read_json(path_report_path)
            archive_prefix = role_dir.relative_to(root).as_posix()
            expected_archive_path = f"{archive_prefix}/{expected_relative}"
            if (not isinstance(path_report, dict) or
                    path_report.get("format") != "KERNELWIKI-KWV08-SANITIZER-PATHS/1" or
                    path_report.get("application_stdout") != expected_archive_path or
                    not isinstance(path_report.get("files"), list) or
                    reported not in {item.get("source_path") for item in path_report.get("files", [])
                                     if isinstance(item, dict) and item.get("archive_path") == expected_archive_path}):
                raise EvidenceError(f"recurrence sanitizer: nested app stdout path report is inconsistent")
            sanitizer_summary = checked_file(base, run.get("sanitizer_summary"),
                                             "recurrence sanitizer classifier summary")
            sanitizer_log = checked_file(base, run.get("sanitizer_log"),
                                         "recurrence sanitizer raw memcheck log")
            summary_sha = _kwv08_archive_hash(archive, sanitizer_summary, role_dir,
                                              "recurrence sanitizer summary")
            log_sha = _kwv08_archive_hash(archive, sanitizer_log, role_dir,
                                          "recurrence sanitizer raw log")
            summary_data = read_json(sanitizer_summary)
            if (not isinstance(summary_data, dict) or
                    sanitizer_summary.relative_to(role_dir).as_posix() != "sanitizer/run/summary.json" or
                    sanitizer_log.relative_to(role_dir).as_posix() != "sanitizer/run/raw.log" or
                    run["sanitizer_summary"].get("sha256") != summary_sha or
                    run["sanitizer_log"].get("sha256") != log_sha or
                    summary_data.get("tool") != "compute-sanitizer" or
                    summary_data.get("status") != "partial" or
                    summary_data.get("tool_detail") != "memcheck" or
                    summary_data.get("conclusive") is not False or
                    type(summary_data.get("exit_status")) is not int or summary_data["exit_status"] != 0 or
                    type(summary_data.get("error_count")) is not int or summary_data["error_count"] != 0):
                raise EvidenceError("recurrence sanitizer: classifier is not the accepted zero-error partial memcheck result")
            raw_log_text = sanitizer_log.read_text(encoding="utf-8", errors="replace")
            error_summaries = [line for line in raw_log_text.splitlines() if "ERROR SUMMARY:" in line]
            reported_log = run["sanitizer_log"].get("reported_path")
            if (len(error_summaries) != 1 or not error_summaries[0].strip().endswith("0 errors") or
                    not isinstance(reported_log, str) or
                    f"raw log: {reported_log}" not in outer_text):
                raise EvidenceError("recurrence sanitizer: raw memcheck capture does not report exactly zero errors")
            report_files = path_report.get("files")
            for relative, reported_path, expected_sha in (
                    ("sanitizer/run/raw.log", reported_log, log_sha),
                    ("sanitizer/run/summary.json", run["sanitizer_summary"].get("reported_path"), summary_sha)):
                archive_path = f"{archive_prefix}/{relative}"
                if not isinstance(reported_path, str) or reported_path not in {
                        item.get("source_path") for item in report_files
                        if isinstance(item, dict) and item.get("archive_path") == archive_path and
                        item.get("sha256") == expected_sha}:
                    raise EvidenceError(f"recurrence sanitizer: nested {relative} path/hash is not captured")
        else:
            profile = run.get("profile")
            if not isinstance(profile, dict):
                raise EvidenceError("recurrence profile: NCU artifact bindings are missing")
            expected_paths = {
                "summary": "ncu/run/summary.json", "analysis": "ncu/run/analysis.txt",
                "raw_csv": "ncu/run/raw.csv", "report": "ncu/run/report.ncu-rep",
                "shim_log": "ncu-shim.jsonl",
            }
            for key, relative in expected_paths.items():
                path = checked_file(base, profile.get(key), f"recurrence NCU {key}")
                if path.relative_to(role_dir).as_posix() != relative:
                    raise EvidenceError(f"recurrence profile: NCU {key} path differs from the captured artifact")
                _kwv08_archive_hash(archive, path, role_dir, f"recurrence NCU {key}")
                if profile[key].get("sha256") != digest(path):
                    raise EvidenceError(f"recurrence profile: NCU {key} hash differs from archive")
                if key == "raw_csv" and path.stat().st_size == 0:
                    raise EvidenceError("recurrence profile: NCU raw counter CSV is empty")
            ncu_summary = read_json(role_dir / expected_paths["summary"])
            if (not isinstance(ncu_summary, dict) or
                    ncu_summary.get("counter_valid") is not True or
                    ncu_summary.get("timing_valid") is not False):
                raise EvidenceError("recurrence profile: NCU counter/timing validity distinction is missing")
    return archive, evidence


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
                                implementations: dict[str, dict[str, Any]],
                                source_root: Path | None = None,
                                portable: bool = False) -> None:
    source_root = source_root or root
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
        if _input_bindings(summary, root, source_root).get(str(reference_path.resolve())) != reference_sha:
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
                                expected_path: Path, source_root: Path,
                                portable: bool) -> None:
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
    if portable:
        mapped = _current_path_for_capture(str(input_arg), root, source_root)
        input_arg = Path(mapped)
    elif not input_arg.is_absolute():
        input_arg = root / input_arg
    if input_arg.resolve() != expected_path.resolve():
        raise EvidenceError(f"{role}: controller argv --input does not resolve to represented GEMM input")


def _validate_gemm_primary_argv(summaries: dict[str, dict[str, Any]], root: Path,
                               source_root: Path, portable: bool,
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
        _validate_gemm_input_option(argv, role, root, input_path, source_root, portable)
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


def _attention_validate_argv(summary: dict[str, Any], root: Path, source_root: Path,
                             portable: bool, archives: dict[str, dict[str, Any]], role: str,
                             expected_impl: str, expected_mode: str, expected_case: str,
                             expected_splits: int,
                             implementation: dict[str, Any], input_path: Path,
                             oracle_path: Path) -> None:
    provenance = summary.get("provenance")
    argv = provenance.get("argv") if isinstance(provenance, dict) else None
    if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
        raise EvidenceError(f"{role}: controller provenance argv is missing")
    executable = Path(argv[0])
    if portable:
        if not executable.is_absolute():
            executable = source_root / executable
    elif not executable.is_absolute():
        executable = root / executable
    binary = provenance.get("binary")
    binary_path = Path(str(binary.get("path", ""))) if isinstance(binary, dict) else Path()
    if not binary_path.is_absolute():
        binary_path = (source_root if portable else root) / binary_path
    if (executable.resolve() != implementation["executed_path"].resolve() or
            binary_path.resolve() != implementation["executed_path"].resolve()):
        raise EvidenceError(f"{role}: controller argv executable differs from the declared attention binary")
    input_identity = Path(_controller_input_identity(input_path, archives))
    oracle_identity = Path(_controller_input_identity(oracle_path, archives))
    try:
        input_selector = input_identity.relative_to(root).as_posix()
        oracle_selector = oracle_identity.relative_to(root).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"{role}: input or reference selector is not repository-relative") from exc
    _attention_selector(argv, role, "--input", input_selector)
    _attention_selector(argv, role, "--reference", oracle_selector)
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


def _attention_validate_profile(root: Path, source_root: Path, portable: bool,
                                base: Path, captured_archives: dict[str, dict[str, Any]],
                                run: dict[str, Any],
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
    shim_relative_path = "benchmark_runs/volta-environment/ncu-collection-shim.py"
    shim_path = _captured_project_file(root, shim_relative_path, captured_archives,
                                       f"{label} checked-in NCU shim")
    shim_sha = digest(shim_path)
    ncu_path = Path(str(ncu_identity["path"])).resolve()
    inputs = _input_bindings(summary, root, source_root)
    historical_shim_path = (source_root / shim_relative_path).resolve()
    shim_identity_path = (root / shim_relative_path).resolve()
    collections = [item for item in shim_records if isinstance(item, dict) and
                   item.get("policy") == "collection-clock-control-none-source-import-yes"]
    if not collections or not all(
        isinstance(item.get("argv_executed"), list) and
        _argument_values(item["argv_executed"], "--clock-control") == ["none"] and
        _argument_values(item["argv_executed"], "--import-source") == ["yes"] and
        isinstance(item.get("shim_path"), str) and Path(item["shim_path"]).resolve() == historical_shim_path and
        item.get("shim_sha256") == shim_sha and inputs.get(str(shim_identity_path)) == shim_sha and
        isinstance(item.get("actual_ncu_path"), str) and
        Path(item["actual_ncu_path"]).resolve() == ncu_path and
        _is_sha256(item.get("actual_ncu_sha256")) and
        (portable or (ncu_path.is_file() and item.get("actual_ncu_sha256") == digest(ncu_path))) and
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
                                   sanitizer_log: Path | None = None,
                                   portable: bool = False) -> None:
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
    lease_path = (checked_file(base, lease_record, f"attention {role} archived lease")
                  if portable else (root / lease_record["path"]).resolve())
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


def _kwv07_suite_records(root: Path, case_id: str, contract_path: Path,
                         contract: dict[str, Any], input_path: Path, input_sha: str,
                         oracle_path: Path, oracle_sha: str) -> dict[str, Any]:
    """Bind a grouped/W4 suite contract to its exact inputs, FP64 files, and CPU precheck."""
    spec = KWV07_ROLE_SPECS.get(case_id)
    frozen = KWV07_FROZEN_SUITES.get(case_id)
    if spec is None or frozen is None:
        raise EvidenceError("KWV07 contract has an unknown aggregate family identity")
    bundle_root = contract_path.parent.parent.resolve()
    try:
        bundle_root.relative_to(root.resolve())
    except ValueError as exc:
        raise EvidenceError("KWV07 fixture bundle is outside the repository") from exc

    if (contract.get("format") != KWV07_SUITE_CONTRACT_FORMAT or
            contract.get("case_id") != case_id or contract.get("family") != spec["family"] or
            contract.get("path_basis") != "bundle root" or
            contract_path.resolve() != (bundle_root / "contracts" / f"{spec['suite']}-suite.json").resolve()):
        raise EvidenceError(f"{case_id}: suite contract identity or path basis is invalid")
    case_ids = contract.get("case_ids")
    expected_case_ids = list(frozen["cases"])
    if (not isinstance(case_ids, list) or not case_ids or any(
            not isinstance(item, str) or not item for item in case_ids)):
        raise EvidenceError(f"{case_id}: ordered suite case IDs are malformed")
    if case_ids != expected_case_ids:
        raise EvidenceError(f"{case_id}: ordered suite case IDs differ from the frozen KWV07 suite")

    suite_record = contract.get("suite_manifest")
    if not isinstance(suite_record, dict):
        raise EvidenceError(f"{case_id}: suite manifest binding is missing")
    suite_raw, suite_sha = suite_record.get("path"), suite_record.get("sha256")
    if (not isinstance(suite_raw, str) or not isinstance(suite_sha, str) or len(suite_sha) != 64):
        raise EvidenceError(f"{case_id}: suite manifest binding is malformed")
    suite_path = (bundle_root / suite_raw).resolve()
    if (not suite_path.is_relative_to(bundle_root) or not suite_path.is_file() or
            digest(suite_path) != suite_sha or suite_sha != frozen["suite_sha256"]):
        raise EvidenceError(f"{case_id}: suite manifest hash/path differs from the frozen contract")
    suite = read_json(suite_path)
    suite_cases = suite.get("cases") if isinstance(suite, dict) else None
    if (not isinstance(suite, dict) or suite.get("format") != KWV07_SUITE_FORMAT or
            suite.get("suite_id") != case_id or suite.get("family") != spec["family"] or
            suite.get("path_basis") != "bundle root" or
            not isinstance(suite_cases, list) or
            any(not isinstance(item, dict) for item in suite_cases) or
            [item.get("id") for item in suite_cases if isinstance(item, dict)] != case_ids):
        raise EvidenceError(f"{case_id}: suite manifest does not match contract case order")
    if (input_path.resolve() != (bundle_root / "bundle-manifest.json").resolve() or
            input_sha != digest(input_path) or input_sha != KWV07_FROZEN_BUNDLE_SHA256 or
            suite.get("source_bundle_manifest") != {"path": "bundle-manifest.json", "sha256": input_sha}):
        raise EvidenceError(f"{case_id}: runner bundle does not match the suite's source manifest")
    bundle = read_json(input_path)
    bundle_cases = bundle.get("cases") if isinstance(bundle, dict) else None
    if not isinstance(bundle, dict) or not isinstance(bundle_cases, list) or any(
            not isinstance(item, dict) for item in bundle_cases):
        raise EvidenceError(f"{case_id}: fixture bundle case records are malformed")
    bundle_by_id = {item.get("id"): item for item in bundle_cases if isinstance(item, dict)}
    if len(bundle_by_id) != len(bundle_cases):
        raise EvidenceError(f"{case_id}: fixture bundle repeats or malforms a case ID")

    if (oracle_path.resolve() != (bundle_root / "oracles" / f"{spec['suite']}-suite.json").resolve() or
            oracle_sha != digest(oracle_path) or oracle_sha != frozen["oracle_sha256"]):
        raise EvidenceError(f"{case_id}: CPU oracle path does not match the aggregate family")
    oracle = read_json(oracle_path)
    oracle_checks = oracle.get("checks") if isinstance(oracle, dict) else None
    if (not isinstance(oracle, dict) or oracle.get("format") != "KERNELWIKI-CPU-ORACLE/1" or
            oracle.get("suite_id") != case_id or oracle.get("passed") is not True or
            oracle.get("validation_scope") !=
            "CPU-only represented-input/reference identity and fixed-policy FP32 simulation precheck; no GPU execution or GPU numerical result is asserted" or
            oracle.get("suite_manifest") != {"path": suite_raw, "sha256": suite_sha} or
            not isinstance(oracle_checks, list) or
            [item.get("case_id") for item in oracle_checks if isinstance(item, dict)] != case_ids):
        raise EvidenceError(f"{case_id}: CPU oracle wrapper is not the matching CPU-only precheck")

    suite_files_path = bundle_root / "suite-files.json"
    if digest(suite_files_path) != KWV07_FROZEN_SUITE_FILES_SHA256:
        raise EvidenceError(f"{case_id}: suite file index differs from the frozen KWV07 bundle")
    suite_files = read_json(suite_files_path)
    files = suite_files.get("files") if isinstance(suite_files, dict) else None
    if (not isinstance(suite_files, dict) or
            suite_files.get("format") != "KERNELWIKI-KWV07-SUITE-FILES/1" or
            not isinstance(files, dict)):
        raise EvidenceError(f"{case_id}: suite file hash index is malformed")
    expected_suite_files = {
        f"contracts/{spec['suite']}-suite.json": digest(contract_path),
        f"oracles/{spec['suite']}-suite.json": oracle_sha,
        f"suites/{spec['suite']}.json": suite_sha,
    }
    for name, expected_sha in expected_suite_files.items():
        if files.get(name) != expected_sha:
            raise EvidenceError(f"{case_id}: suite file index does not bind {name}")
    if (digest(contract_path) != frozen["contract_sha256"] or
            suite.get("source_bundle_manifest", {}).get("sha256") != KWV07_FROZEN_BUNDLE_SHA256):
        raise EvidenceError(f"{case_id}: contract or bundle manifest differs from the frozen KWV07 suite")

    if (len(suite_cases) != len(oracle_checks) or len(case_ids) == 0 or
            any(not isinstance(item, dict) for item in oracle_checks)):
        raise EvidenceError(f"{case_id}: case/oracle arrays are incomplete")
    dimensions: dict[str, dict[str, Any]] = {}
    expected_inputs: dict[str, str] = {}
    expected_references: dict[str, str] = {}
    w4_weights: dict[str, tuple[bytes, bytes, int]] = {}
    total_outputs = 0
    for record, oracle_check in zip(suite_cases, oracle_checks):
        case_name = record["id"]
        if case_name not in bundle_by_id:
            raise EvidenceError(f"{case_id}: {case_name} is absent from the bundle manifest")
        bound_case = bundle_by_id[case_name]
        for field in ("contract", "input", "oracle"):
            if record.get(field) != bound_case.get(field):
                raise EvidenceError(f"{case_id} {case_name}: suite and bundle manifests differ for {field}")
            item = record[field]
            if (not isinstance(item, dict) or not isinstance(item.get("path"), str) or
                    not isinstance(item.get("sha256"), str) or type(item.get("bytes")) is not int):
                raise EvidenceError(f"{case_id} {case_name}: malformed {field} file identity")
            file_path = (bundle_root / item["path"]).resolve()
            if (not file_path.is_relative_to(bundle_root) or not file_path.is_file() or
                    file_path.stat().st_size != item["bytes"] or digest(file_path) != item["sha256"]):
                raise EvidenceError(f"{case_id} {case_name}: {field} bytes differ from the suite manifest")
            if field == "contract":
                per_case = read_json(file_path)
                expected_kind = "grouped_gated" if spec["suite"] == "grouped" else "w4a16"
                shape = per_case.get("shape") if isinstance(per_case, dict) else None
                if (not isinstance(per_case, dict) or per_case.get("kind") != expected_kind or
                        per_case.get("input_sha256") != record["input"]["sha256"] or
                        per_case.get("oracle_sha256") != record["oracle"]["sha256"] or
                        not isinstance(shape, dict)):
                    raise EvidenceError(f"{case_id} {case_name}: case contract differs from its exact inputs")
                expected_case = frozen["cases"].get(case_name)
                if (not isinstance(expected_case, dict) or shape != expected_case["shape"]):
                    raise EvidenceError(f"{case_id} {case_name}: exact dimensions differ from the frozen KWV07 suite")
                if spec["suite"] == "grouped":
                    outputs = shape.get("assignments") * shape.get("n") if all(
                        type(shape.get(field)) is int and shape[field] > 0
                        for field in ("assignments", "n")) else None
                else:
                    outputs = shape.get("m") * shape.get("n") if all(
                        type(shape.get(field)) is int and shape[field] > 0
                        for field in ("m", "n")) else None
                    reuse = per_case.get("reuse_cases")
                    if (not isinstance(reuse, list) or not reuse or
                            any(type(value) is not int or value <= 0 for value in reuse) or
                            reuse != sorted(set(reuse)) or
                            reuse != expected_case["reuse_cases"] or
                            per_case.get("weight_family") != expected_case["weight_family"]):
                        raise EvidenceError(f"{case_id} {case_name}: W4 reuse cases are malformed")
                if type(outputs) is not int or outputs != expected_case["outputs"]:
                    raise EvidenceError(f"{case_id} {case_name}: output count is invalid")
                if oracle_check.get("checked_outputs") != outputs or oracle_check.get("result") != "pass":
                    raise EvidenceError(f"{case_id} {case_name}: CPU oracle output count/result differs")
                numeric = [oracle_check.get("max_abs_error"), oracle_check.get("atol"),
                           oracle_check.get("rtol"), oracle_check.get("max_individual_bound_ratio")]
                if (any(not _finite_nonnegative_number(value) for value in numeric) or
                        oracle_check["rtol"] != 0 or oracle_check["max_abs_error"] > oracle_check["atol"] or
                        oracle_check["max_individual_bound_ratio"] > 1):
                    raise EvidenceError(f"{case_id} {case_name}: CPU oracle tolerance precheck failed")
                dimensions[case_name] = {
                    "outputs": outputs, "atol": float(oracle_check["atol"]),
                    "reuse_cases": per_case.get("reuse_cases", []), "contract": per_case,
                }
                total_outputs += outputs
            elif field == "input":
                expected_inputs[str(file_path)] = item["sha256"]
                if spec["suite"] == "w4a16":
                    per_case = dimensions[case_name]["contract"]
                    shape = per_case["shape"]
                    data = file_path.read_bytes()
                    header = struct.Struct("<8s8I")
                    if len(data) < header.size:
                        raise EvidenceError(f"{case_id} {case_name}: W4 input is shorter than its binary header")
                    magic, kind, m, k, n, experts, group_size, assignments, scale_count = header.unpack_from(data)
                    packed_count = (k * n + 1) // 2
                    codes_start = header.size + 2 * m * k
                    codes_end = codes_start + packed_count
                    scales_count = ((k + group_size - 1) // group_size) * n if group_size else 0
                    if (magic != b"KWV07I01" or kind != 2 or
                            (m, k, n, group_size) != (shape["m"], shape["k"], shape["n"], shape["group_size"]) or
                            experts != 0 or assignments != 0 or scale_count != scales_count or
                            codes_end + 4 * scale_count != len(data)):
                        raise EvidenceError(f"{case_id} {case_name}: W4 input header/length differs from its frozen shape")
                    weight_family = per_case["weight_family"]
                    payload = (data[codes_start:codes_end], data[codes_end:], m)
                    previous = w4_weights.get(weight_family)
                    if previous is not None and (previous[0] != payload[0] or previous[1] != payload[1]):
                        raise EvidenceError(f"{case_id} {case_name}: paired-M codes/scales differ within {weight_family}")
                    w4_weights[weight_family] = payload
            else:
                expected_references[str(file_path)] = item["sha256"]

    if spec["suite"] == "w4a16":
        if set(w4_weights) != {"tail_adversarial", "shared_large_weights"}:
            raise EvidenceError(f"{case_id}: the two frozen paired-M weight families are incomplete")
        for family in w4_weights:
            members = [record for record in dimensions.values()
                       if record["contract"].get("weight_family") == family]
            if (len(members) != 2 or
                    sorted(member["contract"]["shape"]["m"] for member in members) != [1, 16]):
                raise EvidenceError(f"{case_id}: {family} does not provide the frozen M=1/M=16 pair")

    if total_outputs != spec["outputs"]:
        raise EvidenceError(f"{case_id}: frozen aggregate output count differs")
    contract_input_refs = contract.get("input_references")
    contract_oracle_refs = contract.get("oracle_references")
    if (contract_input_refs != [record["input"] for record in suite_cases] or
            contract_oracle_refs != [record["oracle"] for record in suite_cases]):
        raise EvidenceError(f"{case_id}: suite contract does not bind ordered input and FP64 reference lists")
    if len(expected_inputs) != len(case_ids) or len(expected_references) != len(case_ids):
        raise EvidenceError(f"{case_id}: exact suite input/reference set is incomplete")
    return {
        "bundle_root": bundle_root, "suite_path": suite_path, "suite_sha": suite_sha,
        "suite": suite, "suite_files_path": suite_files_path,
        "dimensions": dimensions, "oracle": oracle,
        "expected_inputs": expected_inputs, "expected_references": expected_references,
        "total_outputs": total_outputs,
    }


def _kwv07_finite_metrics(record: dict[str, Any], fields: tuple[str, ...], label: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in fields:
        value = record.get(field)
        if not _finite_nonnegative_number(value):
            raise EvidenceError(f"{label}: {field} is missing, negative, or non-finite")
        values[field] = float(value)
    return values


def _kwv07_samples(record: dict[str, Any], field: str, label: str) -> list[float]:
    values = record.get(field)
    if not isinstance(values, list) or len(values) != 9 or any(
            not _finite_nonnegative_number(value) or float(value) <= 0 for value in values):
        raise EvidenceError(f"{label}: {field} must contain nine positive finite samples")
    return [float(value) for value in values]


def _kwv07_close(actual: Any, expected: float) -> bool:
    return (_finite_nonnegative_number(actual) and
            math.isclose(float(actual), expected, rel_tol=2e-6, abs_tol=1e-9))


def _kwv07_native_checks(result: dict[str, Any], case_id: str, records: dict[str, Any],
                         gpu_uuid: str, role: str, label: str) -> None:
    role_specs = KWV07_ROLE_SPECS[case_id]["roles"]
    role_spec = role_specs.get(role)
    if role_spec is None:
        raise EvidenceError(f"{label}: unknown role")
    _impl_group, implementation, _kind, mode, selected_case = role_spec
    dimensions = records["dimensions"]
    expected_case_ids = list(dimensions) if selected_case is None else [selected_case]
    expected_uuid = gpu_uuid.removeprefix("GPU-").replace("-", "").lower()
    if (result.get("format") != KWV07_NATIVE_FORMAT or result.get("status") != "passed" or
            result.get("suite") != KWV07_ROLE_SPECS[case_id]["suite"] or
            result.get("implementation") != implementation or result.get("mode") != mode or
            result.get("compute_capability") != "7.0" or result.get("gpu_uuid") != expected_uuid or
            not isinstance(result.get("gpu"), str) or not result["gpu"] or
            type(result.get("case_count")) is not int or result["case_count"] != len(expected_case_ids)):
        raise EvidenceError(f"{label}: native suite identity, GPU, implementation, mode, or status differs")
    observed = result.get("cases")
    if (not isinstance(observed, list) or len(observed) != len(expected_case_ids) or
            [item.get("case_id") for item in observed if isinstance(item, dict)] != expected_case_ids):
        raise EvidenceError(f"{label}: native case IDs/order do not match the frozen suite scope")

    top_numeric = _kwv07_finite_metrics(
        result, ("max_abs_error", "max_admission_tolerance", "max_tolerance_ratio"), label)
    for field in ("checked_outputs", "failed_outputs", "nonfinite_outputs"):
        if type(result.get(field)) is not int:
            raise EvidenceError(f"{label}: {field} is not an integer")
    checked = failed = nonfinite = 0
    case_abs: list[float] = []
    case_bounds: list[float] = []
    case_ratios: list[float] = []
    for item, case_name in zip(observed, expected_case_ids):
        expected = dimensions[case_name]
        if type(item.get("checked_outputs")) is not int or item["checked_outputs"] != expected["outputs"]:
            raise EvidenceError(f"{label} {case_name}: checked output count differs from FP64 reference")
        for field in ("failed_outputs", "nonfinite_outputs"):
            if type(item.get(field)) is not int or item[field] != 0:
                raise EvidenceError(f"{label} {case_name}: native failure/non-finite output count is nonzero")
        metrics = _kwv07_finite_metrics(
            item, ("max_abs_error", "max_admission_tolerance", "max_tolerance_ratio"),
            f"{label} {case_name}")
        if (metrics["max_abs_error"] > metrics["max_admission_tolerance"] or
                metrics["max_tolerance_ratio"] > 1 or
                not math.isclose(metrics["max_admission_tolerance"], expected["atol"],
                                 rel_tol=2e-6, abs_tol=1e-12)):
            raise EvidenceError(f"{label} {case_name}: native error or tolerance differs from frozen per-output policy")
        checked += item["checked_outputs"]
        failed += item["failed_outputs"]
        nonfinite += item["nonfinite_outputs"]
        case_abs.append(metrics["max_abs_error"])
        case_bounds.append(metrics["max_admission_tolerance"])
        case_ratios.append(metrics["max_tolerance_ratio"])

        setup = _kwv07_finite_metrics(
            item, ("one_time_setup_host_us", "group_metadata_host_us",
                   "predecode_event_us", "predecode_host_us"), f"{label} {case_name}")
        if setup["group_metadata_host_us"] > setup["one_time_setup_host_us"] + 1e-9:
            raise EvidenceError(f"{label} {case_name}: group metadata is not part of one-time setup")

        if mode == "time":
            events = _kwv07_samples(item, "event_samples_us", f"{label} {case_name}")
            hosts = _kwv07_samples(item, "host_samples_us", f"{label} {case_name}")
            median_event, median_host = statistics.median(events), statistics.median(hosts)
            if (not _kwv07_close(item.get("median_event_us"), median_event) or
                    not _kwv07_close(item.get("median_host_us"), median_host)):
                raise EvidenceError(f"{label} {case_name}: reported timing medians differ from unprofiled samples")
            if case_id == "kwv07-grouped-gated":
                if "reuse_runs" in item or "reuse_assumption" in item or not _kwv07_close(
                        item.get("cold_one_use_host_total_us"), setup["one_time_setup_host_us"] + median_host):
                    raise EvidenceError(f"{label} {case_name}: grouped cold host total/scope differs")
            else:
                if implementation == "direct-w4" and (setup["predecode_event_us"] != 0 or
                                                        setup["predecode_host_us"] != 0):
                    raise EvidenceError(f"{label} {case_name}: direct-W4 route reports predecode work")
                reuse = item.get("reuse_runs")
                uses = expected["reuse_cases"]
                if (item.get("reuse_assumption") != KWV07_REUSE_ASSUMPTION or
                        not isinstance(reuse, list) or
                        [row.get("uses") for row in reuse if isinstance(row, dict)] != uses):
                    raise EvidenceError(f"{label} {case_name}: W4 reuse protocol differs from the per-case contract")
                for row in reuse:
                    row_events = _kwv07_samples(row, "device_event_samples_us", f"{label} {case_name} reuse")
                    row_hosts = _kwv07_samples(row, "compute_host_samples_us", f"{label} {case_name} reuse")
                    cold_event = (setup["predecode_event_us"] if implementation == "predecode-w4" else 0.0) + statistics.median(row_events)
                    cold_host = setup["one_time_setup_host_us"] + setup["predecode_host_us"] + statistics.median(row_hosts)
                    if (not _kwv07_close(row.get("cold_device_event_total_us"), cold_event) or
                            not _kwv07_close(row.get("cold_host_wall_total_us"), cold_host)):
                        raise EvidenceError(f"{label} {case_name} reuse: cold total differs from measured components")
                one_use = next(row for row in reuse if row["uses"] == 1)
                if (one_use["device_event_samples_us"] != item["event_samples_us"] or
                        one_use["compute_host_samples_us"] != item["host_samples_us"]):
                    raise EvidenceError(f"{label} {case_name}: top-level W4 timing does not identify the one-use row")
        elif any(field in item for field in ("event_samples_us", "host_samples_us", "reuse_runs",
                                             "median_event_us", "median_host_us")):
            raise EvidenceError(f"{label} {case_name}: check/profile output contains timing measurements")

    expected_outputs = sum(dimensions[name]["outputs"] for name in expected_case_ids)
    if (checked != expected_outputs or result["checked_outputs"] != expected_outputs or
            result["failed_outputs"] != failed or failed != 0 or
            result["nonfinite_outputs"] != nonfinite or nonfinite != 0 or
            not _kwv07_close(top_numeric["max_abs_error"], max(case_abs)) or
            not _kwv07_close(top_numeric["max_admission_tolerance"], max(case_bounds)) or
            not _kwv07_close(top_numeric["max_tolerance_ratio"], max(case_ratios))):
        raise EvidenceError(f"{label}: native aggregate counts or error maxima differ from case records")


def _kwv07_source_closure(root: Path, case: dict[str, Any], case_id: str,
                          base: Path | None = None, source_root: Path | None = None,
                          portable: bool = False
                          ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    closure_path, closure_sha = sha_binding(root, case.get("source_closure"),
                                            f"{case_id} source closure")
    closure = read_json(closure_path)
    records = closure.get("repository_inputs") if isinstance(closure, dict) else None
    declarations = closure.get("declaring_specs") if isinstance(closure, dict) else None
    if (not isinstance(closure, dict) or
            closure.get("format") != "KERNELWIKI-KWV07-SOURCE-CLOSURE/1" or
            not isinstance(records, list) or not records or
            not isinstance(declarations, list) or len(declarations) != 12):
        raise EvidenceError(f"{case_id}: source closure format or declaring spec set is invalid")
    captured = _captured_input_archives(base or root, root, case, required=True)
    expected: dict[str, tuple[str, str, int]] = {}
    for index, item in enumerate(records):
        label = f"{case_id} source closure input {index}"
        if not isinstance(item, dict):
            raise EvidenceError(f"{label}: malformed repository input")
        original = _repo_relative_identity(root, item.get("original_path"), f"{label} original path")
        archive_raw, expected_sha, size = item.get("archived_path"), item.get("sha256"), item.get("size_bytes")
        if (not isinstance(archive_raw, str) or not isinstance(expected_sha, str) or
                len(expected_sha) != 64 or type(size) is not int or size < 0):
            raise EvidenceError(f"{label}: archive identity is malformed")
        archive_path = absolute_repo_file(root, archive_raw, f"{label} archive")
        if (archive_path.stat().st_size != size or digest(archive_path) != expected_sha or
                str(original) in expected):
            raise EvidenceError(f"{label}: archive hash, size, or original identity differs")
        expected[str(original)] = (str(archive_path), expected_sha, size)
    if set(expected) != set(captured):
        raise EvidenceError(f"{case_id}: captured input aliases differ from source closure")
    for original, (archive, sha, size) in expected.items():
        capture = captured[original]
        if (str(capture["archive_path"]) != archive or capture["sha256"] != sha or
                capture["archive_path"].stat().st_size != size):
            raise EvidenceError(f"{case_id}: captured input alias differs from source closure")

    expected_roles = set().union(*(set(spec["roles"]) for spec in KWV07_ROLE_SPECS.values()))
    expected_specs = {
        f"benchmark_runs/volta-grouped-lowbit/controller/role-{role}.spec.json"
        for role in expected_roles
    }
    declared: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(declarations):
        label = f"{case_id} source closure spec {index}"
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise EvidenceError(f"{label}: malformed spec identity")
        original = _repo_relative_identity(root, item["path"], f"{label} path")
        capture = captured.get(str(original))
        if (capture is None or capture["sha256"] != item.get("sha256") or
                str(original.relative_to(root)) not in expected_specs or str(original) in declared):
            raise EvidenceError(f"{label}: spec is not an exact captured role spec")
        archived_spec = read_json(capture["archive_path"])
        if not isinstance(archived_spec, dict):
            raise EvidenceError(f"{label}: archived spec is not a JSON object")
        role = archived_spec.get("role")
        if role not in expected_roles or role in {record.get("role") for record in declared.values()}:
            raise EvidenceError(f"{label}: role identity is duplicated or unknown")
        inputs = archived_spec.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise EvidenceError(f"{label}: archived role spec input list is missing")
        for raw_input in inputs:
            if not isinstance(raw_input, str) or not raw_input:
                raise EvidenceError(f"{label}: role spec input identity is malformed")
            input_path = Path(raw_input)
            if input_path.is_absolute():
                continue
            input_identity = _repo_relative_identity(root, raw_input, f"{label} declared input")
            if str(input_identity) not in captured:
                raise EvidenceError(f"{label}: declared input is absent from source closure")
        declared[str(original)] = {"role": role, "path": original, "spec": archived_spec}
    if {str(record["path"].relative_to(root)) for record in declared.values()} != expected_specs or {
            record["role"] for record in declared.values()} != expected_roles:
        raise EvidenceError(f"{case_id}: source closure does not declare each exact role spec")
    input_sets = [set(record["spec"]["inputs"]) for record in declared.values()]
    if any(input_set != input_sets[0] for input_set in input_sets[1:]):
        raise EvidenceError(f"{case_id}: role specs do not share the frozen repository input set")
    return {"path": closure_path, "sha256": closure_sha, "data": closure,
            "declared_specs": {record["role"]: record for record in declared.values()}}, captured


def _kwv07_validate_spec(root: Path, case_id: str, role: str,
                         spec_record: dict[str, Any], binary_path: Path,
                         gpu_uuid: str,
                         expected_role: tuple[str, str, str, str, str | None],
                         source_root: Path | None = None) -> None:
    source_root = source_root or root
    _implementation_group, implementation, _kind, mode, selected_case = expected_role
    spec = spec_record["spec"]
    suite = KWV07_ROLE_SPECS[case_id]["suite"]
    expected_recipe = "ncu" if _kind == "foreground-ncu" else (
        "compute-sanitizer" if _kind == "foreground-compute-sanitizer" else "baseline")
    expected_argv = [
        str(binary_path.relative_to(source_root)), "--bundle",
        "benchmark_runs/volta-grouped-lowbit/cases/fixture_bundle",
        "--suite", suite, "--impl", implementation, "--mode", mode,
    ]
    if selected_case is not None:
        expected_argv += ["--case", selected_case]
    if (spec.get("role") != role or spec.get("contract_id") != case_id or
            spec.get("recipe") != expected_recipe or spec.get("argv") != expected_argv or
            spec.get("project_root") != str(source_root) or spec.get("command_cwd") != str(source_root) or
            spec.get("resources", {}).get("gpu_uuids") != [gpu_uuid]):
        raise EvidenceError(f"{case_id} {role}: archived controller spec does not match the native role")
    benchmark = spec.get("benchmark")
    build_argv = benchmark.get("build_argv") if isinstance(benchmark, dict) else None
    first_build_role = role == "grouped_correctness_baseline"
    if first_build_role:
        if (not isinstance(build_argv, list) or not build_argv or
                build_argv[0] != "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc" or
                "-arch=sm_70" not in build_argv or "-lcublas" not in build_argv or
                str((source_root / "artifacts/kernels/volta-grouped-lowbit/variants/kwv07_runner.cu").resolve()) not in build_argv or
                str((source_root / "artifacts/kernels/volta-grouped-lowbit/variants/grouped_lowbit.cu").resolve()) not in build_argv or
                build_argv[-2:] != ["-o", str(binary_path)]):
            raise EvidenceError(f"{case_id} {role}: first grouped role does not build the shared SM70 binary")
    elif build_argv:
        raise EvidenceError(f"{case_id} {role}: shared KWV07 binary was rebuilt after the first role")


def _kwv07_validate_timing_scope(case_id: str, case: dict[str, Any]) -> None:
    expected = KWV07_TIMING_SCOPES.get(case_id)
    if expected is None or case.get("timing_scope") != expected:
        raise EvidenceError(f"{case_id}: timing scope declaration differs from the frozen KWV07 measurement protocol")


def _kwv07_validate_argv(summary: dict[str, Any], root: Path, case_id: str,
                         role: str, implementation: dict[str, Any],
                         expected_role: tuple[str, str, str, str, str | None],
                         source_root: Path | None = None) -> None:
    source_root = source_root or root
    _impl_group, native_impl, _kind, mode, selected_case = expected_role
    provenance = summary.get("provenance")
    argv = provenance.get("argv") if isinstance(provenance, dict) else None
    if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
        raise EvidenceError(f"{case_id} {role}: controller application argv is missing")
    executable = Path(argv[0])
    if not executable.is_absolute():
        executable = source_root / executable
    if executable.resolve() != implementation["executed_path"].resolve():
        raise EvidenceError(f"{case_id} {role}: executed native binary differs from shared ELF")
    expected = [
        "--bundle", "benchmark_runs/volta-grouped-lowbit/cases/fixture_bundle",
        "--suite", KWV07_ROLE_SPECS[case_id]["suite"],
        "--impl", native_impl, "--mode", mode,
    ]
    if selected_case is not None:
        expected += ["--case", selected_case]
    if argv[1:] != expected:
        raise EvidenceError(f"{case_id} {role}: native argv selectors differ from the frozen role")


def _kwv07_archive_hash(manifest: dict[str, Any], path: Path, role_dir: Path,
                        label: str) -> str:
    try:
        relative = path.resolve().relative_to(role_dir.resolve()).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"{label}: archived file is outside its role directory") from exc
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise EvidenceError(f"{label}: role archive file inventory is missing")
    expected = files.get(relative)
    actual = digest(path)
    if not isinstance(expected, str) or expected != actual:
        raise EvidenceError(f"{label}: archive manifest does not bind {relative}")
    return actual


def _kwv07_bind_stdout_capture(base: Path, root: Path, case_id: str,
                               role: str, gpu_uuid: str, run: dict[str, Any], native_stdout: Path,
                               sanitizer_summary: Path | None = None,
                               sanitizer_log: Path | None = None,
                               portable: bool = False) -> dict[str, Any]:
    archive_path = checked_file(base, run.get("archive_manifest"), f"{case_id} {role} archive manifest")
    archive = read_json(archive_path)
    evidence_path = checked_file(base, run.get("evidence"), f"{case_id} {role} controller export")
    exported = read_json(evidence_path)
    evidence = exported.get("evidence") if isinstance(exported, dict) else None
    summary = evidence.get("summary") if isinstance(evidence, dict) else None
    artifacts = evidence.get("artifacts") if isinstance(evidence, dict) else None
    record = summary.get("record") if isinstance(summary, dict) else None
    stdout_path = record.get("stdout") if isinstance(record, dict) else None
    stdout_artifacts = [item for item in artifacts if isinstance(item, dict) and
                        item.get("kind") == "stdout"] if isinstance(artifacts, list) else []
    role_dir = archive_path.parent
    outer = role_dir / "foreground.stdout.txt"
    expected_uuid = gpu_uuid.removeprefix("GPU-").replace("-", "").lower()
    files = archive.get("files") if isinstance(archive, dict) else None
    outer_hash = files.get("foreground.stdout.txt") if isinstance(files, dict) else None
    if (not isinstance(archive, dict) or archive.get("role") != role or
            archive.get("evidence_id") != (evidence.get("id") if isinstance(evidence, dict) else None) or
            archive.get("gpu_uuid_raw_hex") != expected_uuid or
            str(archive.get("compute_capability")) != "7.0" or
            not isinstance(stdout_path, str) or len(stdout_artifacts) != 1 or
            stdout_artifacts[0].get("complete") != 1 or
            stdout_artifacts[0].get("path") != stdout_path or
            stdout_artifacts[0].get("content_hash") != outer_hash or
            not isinstance(outer_hash, str) or len(outer_hash) != 64 or
            not outer.is_file() or digest(outer) != outer_hash):
        raise EvidenceError(f"{case_id} {role}: controller stdout is not hash-bound to the role archive")

    for archive_key, run_key in (("controller-run.json", "execution"),
                                 ("project-control-result.json", "evidence")):
        path = checked_file(base, run.get(run_key), f"{case_id} {role} {run_key}")
        if files.get(archive_key) != digest(path):
            raise EvidenceError(f"{case_id} {role}: archive manifest differs from {archive_key}")
    lease = run.get("lease")
    if not isinstance(lease, dict):
        raise EvidenceError(f"{case_id} {role}: lease capture is missing")
    lease_path = (checked_file(base, lease, f"{case_id} {role} archived lease") if portable
                  else absolute_repo_file(root, lease.get("path"), f"{case_id} {role} lease"))
    if files.get("lease.json") != digest(lease_path) or digest(lease_path) != lease.get("sha256"):
        raise EvidenceError(f"{case_id} {role}: archive manifest differs from lease capture")

    capture = run.get("native_stdout_capture", run.get("native_stdout"))
    capture_path = checked_file(base, capture, f"{case_id} {role} native stdout capture")
    if capture_path.resolve() != native_stdout.resolve():
        raise EvidenceError(f"{case_id} {role}: native stdout capture path differs from declared stdout")
    capture_hash = _kwv07_archive_hash(archive, capture_path, role_dir,
                                       f"{case_id} {role} native stdout")
    if capture.get("sha256") != capture_hash:
        raise EvidenceError(f"{case_id} {role}: native stdout capture hash differs")
    nested = role.endswith("sanitizer_candidate") or role.endswith("profile_candidate")
    if role.endswith("sanitizer_candidate"):
        expected_suffix = "sanitizer/run/stdout.txt"
    elif role.endswith("profile_candidate"):
        expected_suffix = "ncu/run/command.stdout.txt"
    else:
        expected_suffix = "foreground.stdout.txt"
    try:
        capture_suffix = capture_path.resolve().relative_to(role_dir.resolve()).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"{case_id} {role}: native stdout is outside its role archive") from exc
    if nested:
        reported = capture.get("reported_path")
        try:
            outer_text = outer.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise EvidenceError(f"{case_id} {role}: outer controller stdout cannot be read: {exc}") from exc
        marker = "stdout: " if role.endswith("sanitizer_candidate") else "command stdout: "
        if (capture_suffix != expected_suffix or not isinstance(reported, str) or
                not Path(reported).is_absolute() or f"{marker}{reported}" not in outer_text):
            raise EvidenceError(f"{case_id} {role}: nested native stdout path is not identified by captured outer output")
    elif (capture_suffix != expected_suffix or capture.get("reported_path") is not None or
          capture_hash != outer_hash or stdout_artifacts[0].get("content_hash") != capture_hash):
        raise EvidenceError(f"{case_id} {role}: native stdout differs from controller-exported stdout")

    if role.endswith("sanitizer_candidate"):
        if sanitizer_summary is None or sanitizer_log is None:
            raise EvidenceError(f"{case_id} {role}: sanitizer summary and raw log are missing")
        summary_hash = _kwv07_archive_hash(archive, sanitizer_summary, role_dir,
                                           f"{case_id} {role} sanitizer summary")
        log_hash = _kwv07_archive_hash(archive, sanitizer_log, role_dir,
                                       f"{case_id} {role} sanitizer raw log")
        sanitizer_data = read_json(sanitizer_summary)
        if (not isinstance(sanitizer_data, dict) or
                sanitizer_data.get("tool") != "compute-sanitizer" or
                sanitizer_data.get("status") != "partial" or
                sanitizer_data.get("tool_detail") != "memcheck" or
                sanitizer_data.get("conclusive") is not False or
                type(sanitizer_data.get("exit_status")) is not int or sanitizer_data["exit_status"] != 0 or
                type(sanitizer_data.get("error_count")) is not int or sanitizer_data["error_count"] != 0):
            raise EvidenceError(f"{case_id} {role}: sanitizer classifier is not the accepted raw-zero partial memcheck result")
        try:
            raw_text = sanitizer_log.read_text(encoding="utf-8", errors="replace")
            outer_text = outer.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise EvidenceError(f"{case_id} {role}: sanitizer capture cannot be read: {exc}") from exc
        summaries = [line for line in raw_text.splitlines() if "ERROR SUMMARY:" in line]
        reported_log = run.get("sanitizer_log", {}).get("reported_path")
        reported_stdout = capture.get("reported_path")
        if (len(summaries) != 1 or not summaries[0].strip().endswith("0 errors") or
                not isinstance(reported_log, str) or
                f"stdout: {reported_stdout}" not in outer_text or
                f"raw log: {reported_log}" not in outer_text):
            raise EvidenceError(f"{case_id} {role}: raw sanitizer capture does not report exactly zero errors")
        if files.get("sanitizer/run/summary.json") != summary_hash or files.get("sanitizer/run/raw.log") != log_hash:
            raise EvidenceError(f"{case_id} {role}: sanitizer files differ from archive manifest")
    return archive


def _kwv07_bind_profile_files(base: Path, run: dict[str, Any], label: str,
                              archive: dict[str, Any]) -> None:
    profile = run.get("profile")
    if not isinstance(profile, dict):
        raise EvidenceError(f"{label}: NCU profile artifact records are missing")
    archive_path = checked_file(base, run.get("archive_manifest"), f"{label} archive manifest")
    role_dir = archive_path.parent
    for key in ("report", "analysis", "raw_csv", "shim_log"):
        path = checked_file(base, profile.get(key), f"{label} NCU {key}")
        _kwv07_archive_hash(archive, path, role_dir, f"{label} NCU {key}")
        if key == "raw_csv" and path.stat().st_size == 0:
            raise EvidenceError(f"{label}: NCU raw counter CSV is empty")
    if Path(profile["raw_csv"]["path"]).name != "raw.csv":
        raise EvidenceError(f"{label}: NCU raw counter CSV path is not the captured raw.csv artifact")


def _check_kwv07_case(base: Path, root: Path, source_root: Path, portable: bool,
                      case: dict[str, Any], evidence_ids: set[str], contract_path: Path,
                      contract_sha: str, contract: dict[str, Any]) -> None:
    case_id = case["id"]
    role_spec = KWV07_ROLE_SPECS.get(case_id)
    if role_spec is None or contract.get("case_id") != case_id:
        raise EvidenceError("KWV07 case does not identify one of the frozen grouped/W4 suites")
    _kwv07_validate_timing_scope(case_id, case)
    input_path, input_sha = sha_binding(root, case.get("represented_input"),
                                        f"{case_id} represented bundle manifest")
    oracle_path = checked_file(base, case.get("oracle"), f"{case_id} CPU oracle wrapper")
    oracle_sha = digest(oracle_path)
    if case.get("oracle", {}).get("sha256") != oracle_sha:
        raise EvidenceError(f"{case_id}: CPU oracle wrapper binding differs")
    suite_records = _kwv07_suite_records(root, case_id, contract_path, contract,
                                         input_path, input_sha, oracle_path, oracle_sha)
    if case_id not in {"kwv07-grouped-gated", "kwv07-w4a16"}:
        raise EvidenceError(f"{case_id}: unsupported KWV07 family")
    gpu_uuid = case.get("gpu_uuid")
    if (not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-") or
            len(gpu_uuid.removeprefix("GPU-").replace("-", "")) != 32):
        raise EvidenceError(f"{case_id}: declared GPU UUID is malformed")
    uuid_hex = gpu_uuid.removeprefix("GPU-").replace("-", "").lower()

    closure, captured_archives = _kwv07_source_closure(
        root, case, case_id, base=base, source_root=source_root, portable=portable)
    captured_inputs = {str(item["original_path"]): item["sha256"]
                       for item in captured_archives.values()}
    if (str(contract_path.resolve()) not in captured_inputs or
            captured_inputs[str(contract_path.resolve())] != contract_sha or
            str(input_path.resolve()) not in captured_inputs or
            captured_inputs[str(input_path.resolve())] != input_sha or
            str(oracle_path.resolve()) not in captured_inputs or
            captured_inputs[str(oracle_path.resolve())] != oracle_sha):
        raise EvidenceError(f"{case_id}: suite contract, represented bundle, or CPU oracle is absent from source closure")

    implementations = case.get("implementations")
    if not isinstance(implementations, dict):
        raise EvidenceError(f"{case_id}: implementation records are missing")
    bound_impls: dict[str, dict[str, Any]] = {}
    for name in ("baseline", "candidate"):
        entry = implementations.get(name)
        if not isinstance(entry, dict):
            raise EvidenceError(f"{case_id}: {name} implementation record is missing")
        source_path, source_sha = sha_binding(root, entry.get("source"), f"{case_id} {name} source")
        source_record = entry.get("source")
        original_raw = source_record.get("original_path") if isinstance(source_record, dict) else None
        if not isinstance(original_raw, str):
            raise EvidenceError(f"{case_id} {name}: source omits its captured original path")
        original_path = _repo_relative_identity(root, original_raw, f"{case_id} {name} source original path")
        source_capture = captured_archives.get(str(original_path))
        if (source_capture is None or source_capture["archive_path"] != source_path or
                source_capture["sha256"] != source_sha):
            raise EvidenceError(f"{case_id} {name}: source differs from its captured archive")
        binary_path, binary_sha = sha_binding(root, entry.get("binary"), f"{case_id} {name} binary archive")
        executed_raw = entry.get("executed_path")
        if portable:
            executed_relative = _capture_repo_relative(executed_raw, root, source_root)
            if executed_relative is None:
                raise EvidenceError(f"{case_id} {name}: executed binary escapes historical source root")
            executed_path = (source_root / executed_relative).resolve()
        else:
            executed_path = absolute_repo_file(root, executed_raw, f"{case_id} {name} executed binary")
            executed_relative = executed_path.relative_to(root.resolve())
        if not portable and digest(executed_path) != binary_sha:
            raise EvidenceError(f"{case_id} {name}: executed binary differs from shared binary archive")
        bound_impls[name] = {
            "source_path": source_path, "source_input_path": original_path,
            "source_sha": source_sha, "binary_path": binary_path,
            "executed_path": executed_path, "executed_relative": executed_relative,
            "binary_sha": binary_sha,
        }
    baseline, candidate = bound_impls["baseline"], bound_impls["candidate"]
    if (baseline["binary_sha"] != candidate["binary_sha"] or
            baseline["executed_path"] != candidate["executed_path"] or
            baseline["binary_path"] != candidate["binary_path"] or
            baseline["executed_path"].relative_to(source_root).as_posix() !=
            "benchmark_runs/volta-grouped-lowbit/bin/v1/kwv07_runner"):
        raise EvidenceError(f"{case_id}: baseline and candidate do not share the one frozen KWV07 ELF")

    runs = case.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(role_spec["roles"]):
        raise EvidenceError(f"{case_id}: controller receipt set does not match the exact twelve-role plan")
    declared_specs = closure["declared_specs"]
    for role, expected_role in role_spec["roles"].items():
        impl_group, native_impl, expected_kind, mode, selected_case = expected_role
        implementation = bound_impls[impl_group]
        _kwv07_validate_spec(root, case_id, role, declared_specs[role],
                             implementation["executed_path"], gpu_uuid, expected_role,
                             source_root=source_root)

    summaries: dict[str, dict[str, Any]] = {}
    archives: dict[str, dict[str, Any]] = {}
    for role, expected_role in role_spec["roles"].items():
        impl_group, native_impl, expected_kind, mode, selected_case = expected_role
        label = f"{case_id} {role}"
        implementation = bound_impls[impl_group]
        summary = _controller_run(
            base, root, source_root, portable, runs.get(role), label, expected_kind, implementation,
            contract_path, contract_sha, input_path, input_sha, oracle_path, oracle_sha,
            gpu_uuid, evidence_ids, captured_inputs,
            captured_archives=captured_archives,
            allow_duplicate_captured_inputs=set(captured_inputs),
        )
        _kwv07_validate_argv(summary, root, case_id, role, implementation, expected_role,
                             source_root=source_root)
        spec = declared_specs[role]["spec"]
        expected_build = spec.get("benchmark", {}).get("build_argv") if isinstance(
            spec.get("benchmark"), dict) else None
        build = summary.get("build")
        if role == "grouped_correctness_baseline":
            if (not isinstance(build, dict) or build.get("argv") != expected_build or
                    build.get("returncode") != 0 or
                    not isinstance(build.get("stdout_sha256"), str) or
                    not isinstance(build.get("stderr_sha256"), str) or
                    len(build["stdout_sha256"]) != 64 or len(build["stderr_sha256"]) != 64):
                raise EvidenceError(f"{label}: Project Control does not prove the sole successful shared-ELF build")
        elif build is not None:
            raise EvidenceError(f"{label}: shared KWV07 ELF was rebuilt after the first role")

        run = runs[role]
        native_record = run.get("native_stdout_capture", run.get("native_stdout"))
        native_path = checked_file(base, native_record, f"{label} native stdout")
        sanitizer_summary = sanitizer_log = None
        if role.endswith("sanitizer_candidate"):
            sanitizer_summary = checked_file(base, run.get("sanitizer_summary"),
                                             f"{label} Compute Sanitizer summary")
            sanitizer_log = checked_file(base, run.get("sanitizer_log"),
                                         f"{label} raw Compute Sanitizer log")
        archive = _kwv07_bind_stdout_capture(base, root, case_id, role, gpu_uuid,
                                             run, native_path, sanitizer_summary, sanitizer_log,
                                             portable=portable)
        archive_binary_hash = archive.get("files", {}).get("measured_binary")
        if (archive.get("runner_status") != "passed" or
                archive.get("binary_path") not in {
                    str(implementation["executed_path"].relative_to(source_root)),
                    str(implementation["executed_path"]),
                } or archive_binary_hash != implementation["binary_sha"]):
            raise EvidenceError(f"{label}: role archive does not bind the shared native binary or app pass")
        native = read_json(native_path)
        _kwv07_native_checks(native, case_id, suite_records, gpu_uuid, role, label)
        summaries[role] = summary
        archives[role] = archive

    libraries = case.get("libraries")
    if not isinstance(libraries, list):
        raise EvidenceError(f"{case_id}: linked library identities must be an explicit list")
    for library in libraries:
        if (not isinstance(library, dict) or
                library.get("implementation") not in {"baseline", "candidate"} or
                not all(library.get(field) for field in ("name", "version", "path", "sha256"))):
            raise EvidenceError(f"{case_id}: linked library identity/version/path/hash is malformed")
        library_raw = Path(str(library["path"]))
        if portable:
            library_relative = _capture_repo_relative(str(library_raw), root, source_root)
            if library_relative is None:
                library_path = library_raw.resolve()
                if not _is_sha256(library["sha256"]):
                    raise EvidenceError(f"{case_id}: external linked library hash is malformed")
            else:
                library_path = (root / library_relative).resolve()
                archive_record = captured_archives.get(str(library_path))
                if archive_record is not None:
                    if archive_record["sha256"] != library["sha256"]:
                        raise EvidenceError(f"{case_id}: linked library differs from captured repository archive")
                elif (not library_path.is_relative_to(root.resolve()) or not library_path.is_file() or
                      digest(library_path) != library["sha256"]):
                    raise EvidenceError(f"{case_id}: linked repository library is missing or mismatched")
        else:
            library_path = (library_raw if library_raw.is_absolute() else root / library_raw).resolve()
            if not library_path.is_file() or digest(library_path) != library["sha256"]:
                raise EvidenceError(f"{case_id}: linked library path/hash differs")
        for role, summary in summaries.items():
            if _input_bindings(summary, root, source_root).get(str(library_path)) != library["sha256"]:
                raise EvidenceError(f"{case_id} {role}: controller inputs do not bind {library['name']}")

    profile_roles = [role for role, (_impl, _native_impl, kind, _mode, _case) in role_spec["roles"].items()
                     if kind == "foreground-ncu"]
    if len(profile_roles) != 1:
        raise EvidenceError(f"{case_id}: exactly one NCU role must be declared")
    profile_role = profile_roles[0]
    profile = runs[profile_role].get("profile")
    if not isinstance(profile, dict):
        raise EvidenceError(f"{case_id}: selected NCU profile artifacts are missing")
    _attention_validate_profile(root, source_root, portable, base, captured_archives,
                                runs[profile_role], summaries[profile_role],
                                f"{case_id} {profile_role}")
    ncu_archive = archives[profile_role]
    _kwv07_bind_profile_files(base, runs[profile_role], f"{case_id} {profile_role}", ncu_archive)
    if (ncu_archive.get("counter_valid") is not True or
            ncu_archive.get("timing_valid") is not False or
            ncu_archive.get("profile_samples") != 1):
        raise EvidenceError(f"{case_id} {profile_role}: NCU archive must be one counter-valid, timing-invalid profile")


def _check_attention_case(base: Path, root: Path, source_root: Path, portable: bool,
                          case: dict[str, Any], evidence_ids: set[str], contract_path: Path,
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

    captured_archives = _captured_input_archives(base, root, case, required=True)
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
        if portable:
            if not isinstance(executed_raw, str) or not executed_raw or Path(executed_raw).is_absolute():
                raise EvidenceError(f"attention {name}: executed_path must be repository-relative")
            executed_relative = _capture_repo_relative(executed_raw, root, source_root)
            if executed_relative is None:
                raise EvidenceError(f"attention {name}: executed_path escapes historical source_root")
            executed_path = (source_root / executed_relative).resolve()
        else:
            executed_path = absolute_repo_file(root, executed_raw, f"attention {name} executed binary")
            executed_relative = executed_path.relative_to(root.resolve())
        bound_impls[name] = {"source_path": source_path, "source_input_path": source_input_path,
                             "source_original_path": source_original_path,
                             "source_sha": source_sha, "binary_path": binary_path,
                             "executed_path": executed_path, "executed_relative": executed_relative,
                             "binary_sha": binary_sha}

    libraries = case.get("libraries")
    if not isinstance(libraries, list):
        raise EvidenceError("attention linked library identity must be an explicit list")
    bound_libraries = []
    for item in libraries:
        if (not isinstance(item, dict) or item.get("implementation") not in expected_impl_names or
                not all(item.get(key) for key in ("name", "version", "path", "sha256"))):
            raise EvidenceError("attention linked library entry lacks identity/version/path/hash")
        raw_path = Path(str(item["path"]))
        if portable:
            if raw_path.is_absolute():
                try:
                    relative = raw_path.relative_to(source_root)
                except ValueError:
                    try:
                        relative = raw_path.relative_to(root.resolve())
                    except ValueError:
                        if not _is_sha256(item["sha256"]):
                            raise EvidenceError("attention external library SHA-256 is malformed")
                        bound_libraries.append((item, raw_path.resolve()))
                        continue
                if ".." in relative.parts:
                    raise EvidenceError("attention linked library path escapes historical source_root")
                library_path = (root / relative).resolve()
            else:
                library_path = (root / raw_path).resolve()
            archive = captured_archives.get(str(library_path))
            if archive is not None:
                if archive["sha256"] != item["sha256"]:
                    raise EvidenceError("attention linked library differs from its captured archive")
            elif (not library_path.is_relative_to(root.resolve()) or not library_path.is_file() or
                  digest(library_path) != item["sha256"]):
                raise EvidenceError("attention linked library path/hash is unavailable or mismatched")
        else:
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
                base, root, {"id": role, "captured_input_archives": runs[role].get("captured_input_archives")},
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
            base, root, source_root, portable, runs.get(role),
            f"attention {role}", kind, bound_impls[impl_name],
            contract_path, contract_sha, input_path, input_sha, oracle_path, oracle_sha,
            str(case.get("gpu_uuid", "")), evidence_ids, role_captured_inputs,
            additional_inputs=additional_inputs,
            allow_duplicate_captured_inputs=allow_duplicate_captured_inputs,
            captured_archives=role_archives,
        )
        if role in ATTENTION_OPTIONAL_ROLES:
            argv_summ = summary.get("provenance", {}).get("argv", [])
            split_values = _argument_values(argv_summ, "--splits")
            if len(split_values) != 1 or split_values[0] != "16":
                raise EvidenceError(f"attention {role}: supplemental split role must use exactly 16 splits")
            split_count = int(split_values[0])
        else:
            split_count = 1
        _attention_validate_argv(summary, root, source_root, portable, role_archives,
                                 role, native_impl, mode, selected_case,
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
                                       sanitizer_summary_path, sanitizer_log_path,
                                       portable=portable)
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
            if _input_bindings(summaries[role], root, source_root).get(str(library_path)) != library["sha256"]:
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
    _attention_validate_profile(root, source_root, portable, base, captured_archives,
                                runs["profile_candidate_decode"],
                                summaries["profile_candidate_decode"],
                                "attention profile_candidate_decode")


def _check_kwv08_case(base: Path, root: Path, source_root: Path, portable: bool,
                      case: dict[str, Any], evidence_ids: set[str], contract_path: Path,
                      contract_sha: str, contract: dict[str, Any]) -> None:
    case_id = case["id"]
    if case_id != "recurrence_state_ownership_bundle":
        raise EvidenceError("unknown KWV08 recurrence contract id")
    input_path, input_sha = sha_binding(root, case.get("represented_input"),
                                        f"{case_id} represented input")
    oracle_path = checked_file(base, case.get("oracle"), f"{case_id} CPU oracle")
    oracle_sha = digest(oracle_path)
    if case.get("oracle", {}).get("sha256") != oracle_sha:
        raise EvidenceError(f"{case_id}: CPU oracle SHA binding differs")
    records = _kwv08_suite_records(base, root, case, contract_path, contract,
                                   input_path, input_sha, oracle_path, oracle_sha)
    closure, captured_archives, external_inputs = _kwv08_source_closure(
        root, case, base=base, source_root=source_root, portable=portable)
    captured_inputs = {original: item["sha256"] for original, item in captured_archives.items()}

    if case.get("source_closure", {}).get("path") != str(closure["path"].relative_to(root)):
        raise EvidenceError(f"{case_id}: source closure does not use the canonical frozen repository snapshot")
    expected_source = "artifacts/kernels/volta-recurrence/variants/gdn_recurrent_state.cuh"
    expected_launcher = "artifacts/kernels/volta-recurrence/variants/gdn_runner.cu"
    implementations = case.get("implementations")
    if not isinstance(implementations, dict):
        raise EvidenceError(f"{case_id}: baseline/candidate implementation identities are missing")
    executed_path = (source_root / "benchmark_runs/volta-recurrence/bin/gdn_runner").resolve()
    bound_impls: dict[str, dict[str, Any]] = {}
    binary_identity: tuple[Path, str] | None = None
    for name in ("baseline", "candidate"):
        entry = implementations.get(name)
        if not isinstance(entry, dict):
            raise EvidenceError(f"{case_id}: {name} implementation identity is missing")
        source = entry.get("source")
        source_path, source_sha = sha_binding(root, source, f"{case_id} {name} kernel source")
        source_original = _repo_relative_identity(root, source.get("original_path"),
                                                  f"{case_id} {name} source original path")
        source_capture = captured_archives.get(str(source_original))
        if (str(source_original.relative_to(root)) != expected_source or
                source_capture is None or source_path != source_capture["archive_path"] or
                source_sha != source_capture["sha256"]):
            raise EvidenceError(f"{case_id}: {name} source differs from the archived recurrent kernel header")
        launcher = entry.get("launcher_source")
        launcher_path, launcher_sha = sha_binding(root, launcher, f"{case_id} {name} runner source")
        launcher_original = _repo_relative_identity(root, launcher.get("original_path"),
                                                    f"{case_id} {name} runner source original path")
        launcher_capture = captured_archives.get(str(launcher_original))
        if (str(launcher_original.relative_to(root)) != expected_launcher or
                launcher_capture is None or launcher_path != launcher_capture["archive_path"] or
                launcher_sha != launcher_capture["sha256"]):
            raise EvidenceError(f"{case_id}: {name} runner source differs from its archived compiler input")
        binary_path, binary_sha = sha_binding(root, entry.get("binary"),
                                              f"{case_id} {name} measured executable")
        if (not binary_path.name.startswith("gdn_runner-") or
                not binary_path.name.endswith(".elf") or
                binary_path.parent != (base / "evidence/v1/bin").resolve() or
                entry.get("executed_path") != "benchmark_runs/volta-recurrence/bin/gdn_runner" or
                (not portable and (not executed_path.is_file() or digest(executed_path) != binary_sha))):
            raise EvidenceError(f"{case_id}: {name} implementation does not bind the one built shared ELF")
        if binary_identity is None:
            binary_identity = (binary_path, binary_sha)
        elif binary_identity != (binary_path, binary_sha):
            raise EvidenceError(f"{case_id}: baseline and candidate use different native ELF bytes")
        bound_impls[name] = {
            "source_path": source_path, "source_input_path": source_original,
            "source_sha": source_sha, "launcher_source_path": launcher_path,
            "binary_path": binary_path, "executed_path": executed_path,
            "executed_relative": Path("benchmark_runs/volta-recurrence/bin/gdn_runner"),
            "binary_sha": binary_sha,
        }
    if case.get("libraries") != []:
        raise EvidenceError(f"{case_id}: recurrence has no linked math-library comparison inputs")

    runs = case.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(KWV08_ROLES):
        raise EvidenceError(f"{case_id}: exact six controller-backed recurrence roles are required")
    spec_records = closure["declared_specs"]
    summaries: dict[str, dict[str, Any]] = {}
    for role, (kind, implementation_name, _variant, _mode, _selected) in KWV08_ROLES.items():
        if implementation_name not in bound_impls:
            raise EvidenceError(f"recurrence {role}: implementation is not archived")
        spec_record = spec_records.get(f"recurrence_{role}")
        if not isinstance(spec_record, dict):
            raise EvidenceError(f"recurrence {role}: exact source-closed spec is missing")
        _kwv08_validate_spec(root, role, spec_record, executed_path,
                             str(case["gpu_uuid"]), source_root=source_root)
        summary = _controller_run(
            base, root, source_root, portable, runs[role], f"recurrence {role}", kind,
            bound_impls[implementation_name], contract_path, contract_sha,
            input_path, input_sha, oracle_path, oracle_sha,
            str(case["gpu_uuid"]), evidence_ids, captured_inputs,
            captured_archives=captured_archives,
            additional_inputs=external_inputs,
        )
        _kwv08_validate_argv(summary, root, role, bound_impls[implementation_name],
                             source_root=source_root)
        capture = runs[role].get("native_stdout_capture")
        native_path = checked_file(base, capture, f"recurrence {role} native stdout")
        _kwv08_bind_stdout_capture(base, root, role, str(case["gpu_uuid"]),
                                   runs[role], native_path, source_root, portable)
        native_result = read_json(native_path)
        _kwv08_native_checks(native_result, records, str(case["gpu_uuid"]),
                             role, f"recurrence {role}")
        summaries[role] = summary

    if binary_identity is None:
        raise EvidenceError(f"{case_id}: measured ELF identity is unavailable")
    for role, summary in summaries.items():
        role_dir = checked_file(base, runs[role].get("archive_manifest"),
                                f"recurrence {role} archive manifest").parent
        archive = read_json(role_dir / "archive-manifest.json")
        binary = archive.get("binary")
        if (not isinstance(binary, dict) or
                binary.get("path") != str(binary_identity[0].relative_to(root)) or
                binary.get("sha256") != binary_identity[1] or
                not (role_dir / "role-manifest.json").is_file()):
            raise EvidenceError(f"recurrence {role}: archive does not bind the shared measured ELF")

    build_role = "correctness_baseline"
    build_dir = checked_file(base, runs[build_role].get("archive_manifest"),
                             "recurrence build archive manifest").parent
    build_manifest = read_json(build_dir / "archive-manifest.json")
    for name in ("nvcc.stdout.txt", "nvcc.stderr.txt"):
        build_log = build_dir / name
        if not build_log.is_file():
            raise EvidenceError(f"recurrence build: first-build {name} is missing")
        _kwv08_archive_hash(build_manifest, build_log, build_dir, f"recurrence build {name}")
    ptxas = (build_dir / "nvcc.stderr.txt").read_text(encoding="utf-8", errors="replace")
    if ("ptxas info" not in ptxas or "for 'sm_70'" not in ptxas or
            "spill stores" not in ptxas or "Used " not in ptxas):
        raise EvidenceError("recurrence build: captured compiler stderr lacks the SM70 ptxas resource report")


def _check_case(base: Path, root: Path, source_root: Path, portable: bool,
                case: Any, evidence_ids: set[str]) -> None:
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
        _check_attention_case(base, root, source_root, portable, case, evidence_ids,
                              contract, contract_sha, contract_data)
        return
    if _is_kwv07_contract(contract_data):
        _check_kwv07_case(base, root, source_root, portable, case, evidence_ids,
                          contract, contract_sha, contract_data)
        return
    if _is_kwv08_contract(contract_data):
        _check_kwv08_case(base, root, source_root, portable, case, evidence_ids,
                          contract, contract_sha, contract_data)
        return
    rows_contract = _is_rows_contract(contract_data)
    captured_archives = _captured_input_archives(base, root, case, required=rows_contract)
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
        if portable:
            if not isinstance(executed_raw, str) or not executed_raw or Path(executed_raw).is_absolute():
                raise EvidenceError(f"{case['id']} {name} executed_path must be repository-relative")
            executed_relative = _capture_repo_relative(executed_raw, root, source_root)
            if executed_relative is None:
                raise EvidenceError(f"{case['id']} {name} executed_path escapes the historical repository")
            executed_path = (source_root / executed_relative).resolve()
        else:
            executed_path = absolute_repo_file(root, executed_raw, f"{case['id']} {name} executed binary")
            executed_relative = executed_path.relative_to(root.resolve())
        bound_impls[name] = {"source_path": source_path, "source_input_path": source_input_path,
                             "source_sha": source_sha,
                             "binary_path": binary_path, "executed_path": executed_path,
                             "binary_sha": binary_sha, "executed_relative": executed_relative}

    libraries = case.get("libraries")
    if not isinstance(libraries, list):
        raise EvidenceError(f"{case['id']}: linked library identity must be an explicit list (empty if none)")
    bound_libraries = []
    for item in libraries:
        if (not isinstance(item, dict) or
                item.get("implementation") not in {"baseline", "candidate"} or
                not all(isinstance(item.get(key), str) and item.get(key)
                        for key in ("name", "version", "path", "sha256"))):
            raise EvidenceError(f"{case['id']}: linked library entry lacks identity/version/path/hash")
        raw_library_path = Path(item["path"])
        library_sha = item["sha256"]
        if portable:
            if raw_library_path.is_absolute():
                try:
                    relative = raw_library_path.relative_to(source_root)
                except ValueError:
                    try:
                        relative = raw_library_path.relative_to(root.resolve())
                    except ValueError:
                        if not _is_sha256(library_sha):
                            raise EvidenceError(
                                f"{case['id']}: external library identity has a malformed SHA-256"
                            )
                        # External libraries are capture identities; their hashes are bound below
                        # to the original controller inputs, without requiring vendor files locally.
                        library_path = raw_library_path.resolve()
                        bound_libraries.append((item, library_path))
                        continue
                if ".." in relative.parts:
                    raise EvidenceError(f"{case['id']}: library path escapes source_root")
                library_path = (root / relative).resolve()
            else:
                library_path = (root / raw_library_path).resolve()
            archive = captured_archives.get(str(library_path))
            if archive is not None:
                if archive["sha256"] != library_sha:
                    raise EvidenceError(f"{case['id']}: linked library differs from its captured archive")
            elif (not library_path.is_relative_to(root.resolve()) or not library_path.is_file() or
                  digest(library_path) != library_sha):
                raise EvidenceError(f"{case['id']}: linked library archive is unavailable or mismatched")
        else:
            library_path = (raw_library_path if raw_library_path.is_absolute() else root / raw_library_path).resolve()
            if not library_path.is_file() or digest(library_path) != library_sha:
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
            base, root, source_root, portable, runs.get(role), f"{case['id']} {role}", kind,
            bound_impls[implementation_name], contract, contract_sha,
            input_path, input_sha, oracle_path, oracle_sha,
            str(case.get("gpu_uuid", "")), evidence_ids, captured_inputs, captured_archives,
        )
    if _is_gemm_contract(contract_data):
        _validate_gemm_primary_argv(summaries, root, source_root, portable,
                                    contract_data, input_path, bound_impls)
    elif _is_rows_contract(contract_data):
        if rows_reference_path is None or rows_reference_sha is None:
            raise EvidenceError(f"{case['id']}: rows reference binding is missing")
        _validate_rows_primary_argv(
            summaries, root, contract_data, input_path, input_sha,
            rows_reference_path, rows_reference_sha, bound_impls,
            source_root=source_root, portable=portable,
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
            if _input_bindings(summaries[role], root, source_root).get(str(library_path)) != library_sha:
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
    historical_shim_path = (source_root / shim_relative_path).resolve()
    raw_ncu_path = ncu_identity.get("path")
    if not isinstance(raw_ncu_path, str) or not Path(raw_ncu_path).is_absolute():
        raise EvidenceError(f"{case['id']}: NCU tool path must be absolute")
    ncu_path = Path(raw_ncu_path).resolve()
    profile_inputs = _input_bindings(summaries["profile_candidate"], root, source_root)
    if not collection or not all(
        isinstance(item.get("argv_executed"), list) and
        _argument_values(item["argv_executed"], "--clock-control") == ["none"] and
        _argument_values(item["argv_executed"], "--import-source") == ["yes"] and
        isinstance(item.get("shim_path"), str) and Path(item["shim_path"]).resolve() == historical_shim_path and
        item.get("shim_sha256") == shim_sha and
        profile_inputs.get(str(shim_path)) == shim_sha and
        isinstance(item.get("actual_ncu_path"), str) and Path(item["actual_ncu_path"]).resolve() == ncu_path and
        _is_sha256(item.get("actual_ncu_sha256")) and
        (_is_rows_contract(contract_data) or
         profile_inputs.get(str(ncu_path)) == item.get("actual_ncu_sha256")) and
        (portable or (ncu_path.is_file() and item.get("actual_ncu_sha256") == digest(ncu_path)))
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
    portable = "source_root" in manifest
    try:
        source_root = _historical_source_root(manifest.get("source_root"), root, portable)
    except EvidenceError as exc:
        return {"format": "KERNELWIKI-VOLTA-EVIDENCE-CHECK/1", "valid": False,
                "checked_cases": 0, "errors": [str(exc)]}
    evidence_ids: set[str] = set()
    errors: list[str] = []
    checked = 0
    for case in cases:
        try:
            _check_case(path.parent.resolve(), root, source_root, portable, case, evidence_ids)
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
