#!/usr/bin/env python3
"""Export one foreground role and its immutable source/equipment receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


EXPECTED_ROLES = {
    f"recurrence_{role}" for role in (
        "correctness_baseline", "correctness_candidate", "sanitizer_candidate",
        "baseline_timing", "candidate_timing", "profile_candidate",
    )
}
PC_SOURCE = Path("/home/tumlinson/project-control/src")
if str(PC_SOURCE) not in sys.path:
    sys.path.insert(0, str(PC_SOURCE))
from todo_orchestrator.background.store import BackgroundStore  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_file(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise SystemExit(f"required native evidence file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {
        "path": destination.as_posix(),
        "source_path": str(source),
        "sha256": sha256(destination),
        "size_bytes": destination.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--role", required=True, choices=sorted(EXPECTED_ROLES))
    parser.add_argument("--controller-json", type=Path, required=True,
                        help="exact stdout capture from pc_controller.py run --json")
    parser.add_argument("--binary", type=Path, default=None,
                        help="same built runner used by every role; defaults to benchmark_runs/volta-recurrence/bin/gdn_runner")
    parser.add_argument("--ncu-shim-log", type=Path, default=None,
                        help="required for the profile role; exact log passed to pc_controller.py")
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    run_root = repo / "benchmark_runs/volta-recurrence"
    evidence_root = run_root / "evidence/v1"
    role_dir = evidence_root / "roles" / args.role
    if role_dir.exists():
        raise SystemExit(f"refusing to overwrite role evidence: {role_dir}")
    cli_path = args.controller_json.resolve(strict=True)
    controller = json.loads(cli_path.read_text(encoding="utf-8"))
    if not isinstance(controller, dict):
        raise SystemExit("controller JSON must be one object")
    evidence_id = str(controller.get("evidence_id") or "")
    if not evidence_id:
        raise SystemExit("controller JSON has no evidence_id; preserve its failure separately before continuing")
    if args.role == "recurrence_profile_candidate" and args.ncu_shim_log is None:
        raise SystemExit("profile evidence requires the no-clock/source-import NCU shim log")

    # create=False is intentional: this is a read-only export of controller-owned state.
    store = BackgroundStore(repo, create=False)
    result = store.result(evidence_id)
    if result is None:
        raise SystemExit(f"Project Control has no read-only result for evidence ID {evidence_id}")
    snapshot = result.get("snapshot") if isinstance(result.get("snapshot"), dict) else {}
    source_root = Path(str(snapshot.get("source_root", "")))
    fingerprint = str(snapshot.get("fingerprint", ""))
    if not fingerprint or not source_root.is_dir():
        raise SystemExit("Project Control result is missing its materialized source snapshot")

    role_dir.mkdir(parents=True)
    files: list[dict[str, object]] = []
    files.append(copy_file(cli_path, role_dir / "controller-run.json"))
    # Preserve both the exact BackgroundStore result and the wrapper shape used
    # by the shared KWV evidence validator. Current BackgroundStore.result()
    # returns the item directly; keeping both files makes that boundary clear.
    write_json(role_dir / "project-control-read-export.json", result)
    write_json(role_dir / "controller-evidence.json", {"evidence": result})

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise SystemExit("Project Control result has no artifact list")
    copied_kinds: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        kind = str(artifact.get("kind", f"artifact-{index}"))
        source = Path(str(artifact.get("path", "")))
        if not source.is_file():
            raise SystemExit(f"Project Control artifact is missing: {source}")
        safe_kind = "stdout" if kind == "stdout" else "stderr" if kind == "stderr" else f"{kind}-{index}"
        files.append(copy_file(source, role_dir / f"foreground.{safe_kind}.txt"))
        copied_kinds.add(kind)
    if not {"stdout", "stderr"}.issubset(copied_kinds):
        raise SystemExit("foreground stdout and stderr receipts are both required")

    lease_path = Path(str(controller.get("lease_receipt", "")))
    files.append(copy_file(lease_path, role_dir / "lease.json"))

    if args.role == "recurrence_correctness_baseline":
        for name in ("nvcc.stdout.txt", "nvcc.stderr.txt"):
            build_log = evidence_root / "build" / name
            files.append(copy_file(build_log, role_dir / name))

    shared_snapshots = evidence_root / "source-snapshots"
    snapshot_destination = shared_snapshots / fingerprint
    if not snapshot_destination.exists():
        snapshot_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root, snapshot_destination, symlinks=True)
    snapshot_meta = {
        "fingerprint": fingerprint,
        "source_root_from_project_control": str(source_root),
        "archived_source_root": snapshot_destination.relative_to(repo).as_posix(),
        "commit": snapshot.get("commit"),
        "patch_hash": snapshot.get("patch_hash"),
        "untracked": snapshot.get("untracked", []),
    }
    write_json(role_dir / "source-snapshot-reference.json", snapshot_meta)

    binary = (args.binary or run_root / "bin/gdn_runner").resolve()
    binary_record: dict[str, object] = {"available": binary.is_file()}
    if binary.is_file():
        binary_hash = sha256(binary)
        archive_dir = evidence_root / "bin"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"gdn_runner-{binary_hash}.elf"
        if archive.exists() and sha256(archive) != binary_hash:
            raise SystemExit("archived ELF path exists with different bytes")
        if not archive.exists():
            copy_file(binary, archive)
        identity_path = archive_dir / "gdn_runner.identity.json"
        identity = {"path": archive.relative_to(repo).as_posix(), "sha256": binary_hash,
                    "size_bytes": binary.stat().st_size}
        if identity_path.exists():
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
            if existing != identity:
                raise SystemExit("built runner changed; all roles must reuse the exact first-build ELF")
        else:
            write_json(identity_path, identity)
        binary_record = {"available": True, **identity}
    elif args.role != "recurrence_correctness_baseline":
        raise SystemExit("runner ELF is missing after the baseline build role")

    if args.role == "recurrence_profile_candidate":
        shim_log = args.ncu_shim_log.resolve(strict=True)
        files.append(copy_file(shim_log, role_dir / "ncu-shim.jsonl"))
        ncu_dir = lease_path.parent / "ncu"
        if not ncu_dir.is_dir():
            raise SystemExit(f"NCU report directory is missing: {ncu_dir}")
        archived_ncu = role_dir / "ncu"
        shutil.copytree(ncu_dir, archived_ncu)

    if args.role == "recurrence_sanitizer_candidate":
        sanitizer_dir = lease_path.parent / "sanitizer" / "run"
        expected = ("stdout.txt", "raw.log", "summary.json", "summary.txt", "stderr.txt")
        missing = [name for name in expected if not (sanitizer_dir / name).is_file()]
        if missing:
            raise SystemExit(
                "Compute Sanitizer nested evidence is incomplete; missing "
                + ", ".join(missing)
                + f" under {sanitizer_dir}"
            )
        archived_sanitizer = role_dir / "sanitizer" / "run"
        shutil.copytree(sanitizer_dir, archived_sanitizer)
        nested_records = []
        for source in sorted(p for p in sanitizer_dir.iterdir() if p.is_file()):
            destination = archived_sanitizer / source.name
            nested_records.append({
                "source_path": str(source),
                "archive_path": destination.relative_to(repo).as_posix(),
                "sha256": sha256(destination),
                "size_bytes": destination.stat().st_size,
            })
            files.append({
                "path": destination.relative_to(repo).as_posix(),
                "source_path": str(source),
                "sha256": sha256(destination),
                "size_bytes": destination.stat().st_size,
            })
        path_report = role_dir / "sanitizer-path-report.json"
        write_json(path_report, {
            "format": "KERNELWIKI-KWV08-SANITIZER-PATHS/1",
            "application_stdout": (archived_sanitizer / "stdout.txt").relative_to(repo).as_posix(),
            "raw_log": (archived_sanitizer / "raw.log").relative_to(repo).as_posix(),
            "summary_json": (archived_sanitizer / "summary.json").relative_to(repo).as_posix(),
            "summary_text": (archived_sanitizer / "summary.txt").relative_to(repo).as_posix(),
            "files": nested_records,
        })
        files.append({
            "path": path_report.relative_to(repo).as_posix(),
            "source_path": str(sanitizer_dir),
            "sha256": sha256(path_report),
            "size_bytes": path_report.stat().st_size,
        })

    write_json(role_dir / "role-manifest.json", {
        "format": "KERNELWIKI-KWV08-NATIVE-ROLE/1",
        "role": args.role,
        "evidence_id": evidence_id,
        "controller_status": controller.get("status"),
        "controller_returncode": controller.get("returncode"),
        "controller_result_status": result.get("status"),
        "controller_result_summary": result.get("summary"),
        "source_fingerprint": fingerprint,
        "binary": binary_record,
        "copied_files": files,
    })
    print(role_dir.relative_to(repo))


if __name__ == "__main__":
    main()
