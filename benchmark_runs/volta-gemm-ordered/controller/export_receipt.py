#!/usr/bin/env python3
"""Export one already-recorded controller result via read-only BackgroundStore."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from todo_orchestrator.background.store import BackgroundStore


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    execution_path = args.execution.resolve()
    out_dir = args.out_dir.resolve()
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    evidence_id = execution.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise SystemExit("controller execution has no evidence ID")

    result = BackgroundStore(root, create=False).result(evidence_id)
    if result is None:
        raise SystemExit("read-only BackgroundStore result is missing")
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = out_dir / "evidence.json"
    evidence_path.write_text(json.dumps({"evidence": result}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lease_source = Path(str(execution.get("lease_receipt", ""))).resolve()
    if not lease_source.is_file() or not lease_source.is_relative_to(root):
        raise SystemExit("controller lease receipt is unavailable or outside the repository")
    lease_path = out_dir / "lease.json"
    shutil.copyfile(lease_source, lease_path)

    archived_outputs: dict[str, dict[str, str]] = {}
    for artifact in result.get("artifacts", []):
        kind = artifact.get("kind")
        source = Path(str(artifact.get("path", "")))
        if kind not in {"stdout", "stderr"} or not source.is_file():
            continue
        target = out_dir / ("target.stdout.txt" if kind == "stdout" else "target.stderr.txt")
        shutil.copyfile(source, target)
        archived_outputs[kind] = {"path": target.name, "sha256": digest(target)}

    spec_source = args.spec.resolve()
    if not spec_source.is_file():
        raise SystemExit(f"role specification is missing: {spec_source}")
    spec_value = json.loads(spec_source.read_text(encoding="utf-8"))
    binaries = []
    for relative in spec_value.get("binary_paths", []):
        binary = (root / str(relative)).resolve()
        if not binary.is_relative_to(root) or not binary.is_file():
            raise SystemExit(f"declared role binary is missing or outside the repository: {relative}")
        binaries.append({"path": str(binary.relative_to(root)), "sha256": digest(binary)})
    receipt = {
        "format": "KERNELWIKI-VOLTA-ORDERED-ROLE-RECEIPT/1",
        "role": args.role,
        "evidence_id": evidence_id,
        "controller_status": execution.get("status"),
        "controller_ok": execution.get("ok"),
        "controller_returncode": execution.get("returncode"),
        "evidence_status": result.get("status"),
        "evidence_valid": bool(result.get("valid")),
        "evidence_contaminated": bool(result.get("contaminated")),
        "spec": {"path": str(spec_source), "sha256": digest(spec_source)},
        "execution": {"path": execution_path.name, "sha256": digest(execution_path)},
        "evidence": {"path": evidence_path.name, "sha256": digest(evidence_path)},
        "lease": {"path": lease_path.name, "sha256": digest(lease_path), "runtime_path": str(lease_source)},
        "binaries": binaries,
        "target_output": archived_outputs,
    }
    build_manifest = root / "benchmark_runs/volta-gemm-ordered/evidence/shared/runner-build-source-9ab1ba79270f128d5b3cbbd56da8db991799326604ef87973085981672dd3e9c.json"
    if build_manifest.is_file():
        receipt["build_manifest"] = {
            "path": str(build_manifest.relative_to(root)),
            "sha256": digest(build_manifest),
        }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
