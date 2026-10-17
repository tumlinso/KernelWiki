#!/usr/bin/env python3
"""Freeze exact KWV-07 repository inputs and identify the selected toolchain."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = Path("benchmark_runs/volta-grouped-lowbit")
OUT = ROOT / RUN / "evidence/v1/source-closure"
MANIFEST = OUT / "source-closure.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    spec_paths = sorted((ROOT / RUN / "controller").glob("role-*.spec.json"))
    if len(spec_paths) != 12:
        raise SystemExit(f"expected exactly 12 role specs, found {len(spec_paths)}")
    specs = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in spec_paths]
    role_names = {spec.get("role") for _, spec in specs}
    if len(role_names) != 12 or None in role_names:
        raise SystemExit("role spec names must be 12 distinct nonempty roles")
    input_sets = [set(spec.get("inputs", [])) for _, spec in specs]
    if any(not paths for paths in input_sets) or any(paths != input_sets[0] for paths in input_sets[1:]):
        raise SystemExit("all role specs must declare the same nonempty input set")
    expected_roles = {f"{family}_{name}" for family in ("grouped", "w4a16")
                      for name in ("correctness_baseline", "correctness_candidate",
                                   "sanitizer_candidate", "baseline_timing",
                                   "candidate_timing", "profile_candidate")}
    if role_names != expected_roles:
        raise SystemExit("role spec set differs from the fixed KWV-07 role plan")
    declarations = []
    for path, spec in specs:
        if path.is_symlink():
            raise SystemExit(f"role spec path must not be a symlink: {path}")
        declarations.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
        })

    repo_inputs: list[str] = []
    external_inputs: list[str] = []
    for raw in sorted(input_sets[0]):
        path = Path(raw)
        if path.is_absolute():
            external_inputs.append(raw)
        else:
            resolved = (ROOT / path).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.is_file():
                raise SystemExit(f"repository input is missing or escapes root: {raw}")
            repo_inputs.append(path.as_posix())
    if not repo_inputs:
        raise SystemExit("no repository inputs are declared")

    if MANIFEST.exists() or (OUT / "repo").exists():
        raise SystemExit(f"refusing to replace existing source closure: {OUT}")
    (OUT / "repo").mkdir(parents=True)
    frozen: list[dict[str, object]] = []
    for original in repo_inputs:
        src = ROOT / original
        archived = OUT / "repo" / original
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, archived)
        source_sha = sha256(src)
        if sha256(archived) != source_sha or archived.stat().st_size != src.stat().st_size:
            raise SystemExit(f"source closure copy did not preserve {original}")
        frozen.append({
            "original_path": original,
            "archived_path": archived.relative_to(ROOT).as_posix(),
            "sha256": source_sha,
            "size_bytes": src.stat().st_size,
        })

    tools: list[dict[str, object]] = []
    for raw in external_inputs:
        path = Path(raw)
        if not path.is_file():
            raise SystemExit(f"toolchain input is missing: {raw}")
        tools.append({
            "path": raw,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })

    version_commands = {
        "nvcc": ["/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc", "--version"],
        "compute_sanitizer": [
            "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer",
            "--version",
        ],
        "ncu": [
            "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/profilers/12.9/Nsight_Compute/target/linux-desktop-glibc_2_11_3-x64/ncu",
            "--version",
        ],
    }
    versions = {}
    for name, argv in version_commands.items():
        result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=60)
        versions[name] = {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

    manifest = {
        "format": "KERNELWIKI-KWV07-SOURCE-CLOSURE/1",
        "project_root": str(ROOT),
        "repository_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
        "gpu_uuid_target": "GPU-21131915-1488-23af-38dd-1743ae1f5cc8",
        "declaring_specs": declarations,
        "repository_inputs": frozen,
        "toolchain_inputs": tools,
        "tool_versions": versions,
        "binary_policy": "one sm_70 ELF built by grouped_correctness_baseline and reused byte-identically by all later roles",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"froze {len(frozen)} repository inputs and {len(tools)} external toolchain files")
    print(f"manifest: {MANIFEST.as_posix()} sha256={sha256(MANIFEST)}")


if __name__ == "__main__":
    main()
