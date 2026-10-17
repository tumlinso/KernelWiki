#!/usr/bin/env python3
"""Generate the six fixed KWV-08 native ownership comparison roles."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("KWV08_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
RUN = "benchmark_runs/volta-recurrence"
RUN_DIR = Path(__file__).resolve().parents[1]
ART = "artifacts/kernels/volta-recurrence/variants"
SDK = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1"
CUDA = f"{SDK}/cuda/12.9"
CUDA_SKILLS = "/home/tumlinson/.agents/skills/cuda/scripts"
NVCC = f"{CUDA}/bin/nvcc"
SANITIZER = f"{CUDA}/compute-sanitizer/compute-sanitizer"
NCU_SHIM = f"{SDK}/profilers/12.9/Nsight_Compute/ncu"
NCU_REAL = f"{SDK}/profilers/12.9/Nsight_Compute/target/linux-desktop-glibc_2_11_3-x64/ncu"
RUNNER = f"{RUN}/bin/gdn_runner"
BUNDLE = f"{RUN}/inputs/recurrence_inputs.bin"
GPU_UUID = "GPU-21131915-1488-23af-38dd-1743ae1f5cc8"
ROLE_NAMES = (
    "recurrence_correctness_baseline",
    "recurrence_correctness_candidate",
    "recurrence_sanitizer_candidate",
    "recurrence_baseline_timing",
    "recurrence_candidate_timing",
    "recurrence_profile_candidate",
)


def declared_inputs() -> list[str]:
    paths = [
        f"{ART}/PROVENANCE.yaml",
        f"{ART}/README.md",
        f"{ART}/gdn_runner.cu",
        f"{ART}/gdn_recurrent_state.cuh",
        f"{ART}/bundle_contract.hpp",
        f"{ART}/sha256.hpp",
        f"{ART}/tests/test_bundle_reader.cpp",
        f"{RUN}/contract.json",
        f"{RUN}/cpu-oracle.json",
        f"{RUN}/case_matrix.json",
        f"{RUN}/recurrence_inputs.json",
        f"{RUN}/inputs/recurrence_inputs.bin",
        f"{RUN}/source/recurrence_oracle.py",
        f"{RUN}/source/bundle_io.py",
        f"{RUN}/source/prepare_recurrence_bundle.py",
        f"{RUN}/source/probe_fp32_policy.py",
        f"{RUN}/source/test_recurrence_oracle.py",
        f"{RUN}/source/test_recurrence_bundle.py",
        f"{RUN}/source/make_case_records.py",
        f"{RUN}/controller/make_role_specs.py",
        f"{RUN}/source/pack_source_closure.py",
        f"{RUN}/source/collect_native_evidence.py",
        f"{RUN}/source/capture_nvcc_build.sh",
        "benchmark_runs/volta-gemm/controller/pc_controller.py",
        "benchmark_runs/volta-environment/ncu-collection-shim.py",
        f"{CUDA}/compute-sanitizer/compute-sanitizer",
        f"{CUDA_SKILLS}/profile_ncu.sh",
        f"{CUDA_SKILLS}/debug_compute_sanitizer.sh",
        f"{CUDA_SKILLS}/cuda_controller.py",
        f"{CUDA_SKILLS}/cuda_toolchain.py",
        f"{SDK}/cuda/12.9/bin/nvcc",
        f"{SDK}/profilers/12.9/Nsight_Compute/ncu",
        NCU_REAL,
    ]
    paths.extend(f"{RUN}/controller/role-{name}.spec.json" for name in ROLE_NAMES)
    return sorted(set(paths))


def build_argv() -> list[str]:
    return [
        str(ROOT / RUN / "source/capture_nvcc_build.sh"),
        str(ROOT / RUN / "evidence/v1/build/nvcc.stdout.txt"),
        str(ROOT / RUN / "evidence/v1/build/nvcc.stderr.txt"),
        "--",
        NVCC,
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-arch=sm_70",
        "-Xptxas=-v",
        "-Xcompiler=-Wall,-Wextra",
        "-Xcompiler=-fno-omit-frame-pointer",
        str(ROOT / ART / "gdn_runner.cu"),
        "-o",
        str(ROOT / RUNNER),
    ]


def role_config() -> list[tuple[str, str, str, str, str | None, bool]]:
    return [
        (ROLE_NAMES[0], "per_token", "baseline", "correctness", None, True),
        (ROLE_NAMES[1], "retained", "baseline", "correctness", None, False),
        (ROLE_NAMES[2], "retained", "compute-sanitizer", "correctness", None, False),
        (ROLE_NAMES[3], "per_token", "baseline", "time", None, False),
        (ROLE_NAMES[4], "retained", "baseline", "time", None, False),
        (ROLE_NAMES[5], "retained", "ncu", "profile", "long_prefill_many_sequences", False),
    ]


def make_spec(role: str, variant: str, recipe: str, mode: str,
              case: str | None, first_build: bool) -> dict[str, object]:
    argv = [RUNNER, "--bundle", BUNDLE, "--variant", variant, "--mode", mode]
    if case:
        argv.extend(("--case", case))
    snapshot_paths = [raw for raw in declared_inputs() if not Path(raw).is_absolute()]
    snapshot_paths.append(RUNNER)
    spec: dict[str, object] = {
        "schema_version": 1,
        "project_root": str(ROOT),
        "command_cwd": str(ROOT),
        "paths": snapshot_paths,
        "inputs": declared_inputs(),
        "binary_paths": [RUNNER],
        "contract_id": "kwv08-recurrence-state-ownership",
        "toolchain": {"root": CUDA, "require_sanitizer": True},
        "resources": {
            "gpus": 1,
            "gpu_uuids": [GPU_UUID],
            "cpu_threads": 2,
            "isolate_nvlink_domain": False,
            "isolate_pcie_root": False,
        },
        "quiescence": {
            "consecutive_idle_samples": 3,
            "interval_seconds": 0.25,
            "timeout_seconds": 60,
        },
        "role": role,
        "timeout": 7200 if recipe == "ncu" else 3600,
        "recipe": recipe,
        "argv": argv,
    }
    if first_build:
        spec["benchmark"] = {"build_argv": build_argv()}
    return spec


def main() -> None:
    controller = RUN_DIR / "controller"
    controller.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "bin").mkdir(parents=True, exist_ok=True)
    for role, variant, recipe, mode, case, first_build in role_config():
        path = controller / f"role-{role}.spec.json"
        path.write_text(json.dumps(make_spec(role, variant, recipe, mode, case, first_build),
                                    indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(RUN_DIR.parents[2]))


if __name__ == "__main__":
    main()
