#!/usr/bin/env python3
"""Freeze every declared KWV-08 input before documentation edits or GPU runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[3]
RUN = Path("benchmark_runs/volta-recurrence")
OUT = ROOT / RUN / "evidence/v1/source-closure"
MANIFEST = OUT / "source-closure.json"
GPU_UUID = "GPU-21131915-1488-23af-38dd-1743ae1f5cc8"
EXPECTED_ROLES = {
    f"recurrence_{role}" for role in (
        "correctness_baseline", "correctness_candidate", "sanitizer_candidate",
        "baseline_timing", "candidate_timing", "profile_candidate",
    )
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_file(root: Path, raw: str) -> dict[str, object]:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"declared input is missing or is a symlink: {raw}")
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def main() -> None:
    spec_paths = sorted((ROOT / RUN / "controller").glob("role-*.spec.json"))
    if len(spec_paths) != 6:
        raise SystemExit(f"expected six role specs, found {len(spec_paths)}")
    specs = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in spec_paths]
    roles = {str(spec.get("role", "")) for _, spec in specs}
    if roles != EXPECTED_ROLES:
        raise SystemExit(f"unexpected role set: {sorted(roles)}")
    input_sets = [set(spec.get("inputs", [])) for _, spec in specs]
    if any(not values for values in input_sets) or any(values != input_sets[0] for values in input_sets[1:]):
        raise SystemExit("all six role specs must declare the same nonempty input set")
    first_build = [spec for _, spec in specs if "benchmark" in spec and "build_argv" in spec["benchmark"]]
    if len(first_build) != 1 or first_build[0].get("role") != "recurrence_correctness_baseline":
        raise SystemExit("only baseline correctness may own the sole native build")
    binaries = {tuple(spec.get("binary_paths", [])) for _, spec in specs}
    if binaries != {("benchmark_runs/volta-recurrence/bin/gdn_runner",)}:
        raise SystemExit("all roles must use the same native executable path")

    declarations = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
        for path, _ in specs
    ]
    repository_inputs: list[dict[str, object]] = []
    external_inputs: list[dict[str, object]] = []
    for raw in sorted(input_sets[0]):
        entry = hash_file(ROOT, raw)
        if Path(raw).is_absolute():
            external_inputs.append(entry)
        else:
            original = Path(raw)
            archived = OUT / "repo" / original
            repository_inputs.append({
                "original_path": original.as_posix(),
                "archived_path": (Path("benchmark_runs/volta-recurrence/evidence/v1/source-closure/repo") / original).as_posix(),
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            })

    if OUT.exists():
        raise SystemExit(f"refusing to overwrite an existing source closure: {OUT}")
    (OUT / "repo").mkdir(parents=True)
    for item in repository_inputs:
        original = ROOT / str(item["original_path"])
        archived = OUT / "repo" / str(item["original_path"])
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, archived)
        if sha256(archived) != item["sha256"] or archived.stat().st_size != item["size_bytes"]:
            raise SystemExit(f"source closure copy failed verification: {item['original_path']}")

    versions = {}
    commands = {
        "nvcc": ["/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc", "--version"],
        "compute_sanitizer": [
            "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer", "--version"
        ],
        "ncu": [
            "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/profilers/12.9/Nsight_Compute/target/linux-desktop-glibc_2_11_3-x64/ncu",
            "--version",
        ],
    }
    for name, argv in commands.items():
        result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=60)
        versions[name] = {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    manifest = {
        "format": "KERNELWIKI-KWV08-SOURCE-CLOSURE/1",
        "project_root": str(ROOT),
        "repository_revision": revision,
        "gpu_uuid_target": GPU_UUID,
        "declaring_specs": declarations,
        "repository_inputs": repository_inputs,
        "toolchain_inputs": external_inputs,
        "tool_versions": versions,
        "binary_policy": "one sm_70 ELF built by recurrence_correctness_baseline and reused byte-identically by all later roles",
        "timing_policy": "per-token-kernel-launch baseline versus one fused retained-state launch; full sequence cost, host enqueue, and host completion are reported separately; no isolated traffic-causality claim",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"froze {len(repository_inputs)} repository inputs and {len(external_inputs)} toolchain inputs")
    print(f"source closure SHA-256: {sha256(MANIFEST)}")


if __name__ == "__main__":
    main()
