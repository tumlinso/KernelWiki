#!/usr/bin/env python3
"""Write the fixed KWV-07 CUDA controller role specifications.

These JSON files bind one shared native runner to two operator suites.  Only
the first grouped correctness role has a build recipe; later roles execute the
same binary bytes and include no build action.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = "benchmark_runs/volta-grouped-lowbit"
ART = "artifacts/kernels/volta-grouped-lowbit"
SDK = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1"
CUDA = f"{SDK}/cuda/12.9"
CUBLAS = f"{SDK}/math_libs/12.9/targets/x86_64-linux"
NVCC = f"{CUDA}/bin/nvcc"
RUNNER = f"{RUN}/bin/v1/kwv07_runner"
BUNDLE = f"{RUN}/cases/fixture_bundle"
GPU_UUID = "GPU-21131915-1488-23af-38dd-1743ae1f5cc8"


def declared_inputs() -> list[str]:
    paths = [
        f"{ART}/variants/PROVENANCE.yaml",
        f"{ART}/variants/README.md",
        f"{ART}/variants/Makefile.cuda",
        f"{ART}/variants/kwv07_runner.cu",
        f"{ART}/variants/grouped_lowbit.cu",
        f"{ART}/variants/grouped_lowbit.cuh",
        f"{ART}/variants/include/kwv07_bundle.hpp",
        f"{ART}/variants/tests/test_bundle_reader.cpp",
        f"{ART}/variants/tests/test_bundle_reader_errors.cpp",
        f"{RUN}/README.md",
        f"{RUN}/cases/case-matrix.json",
        f"{RUN}/source/README.md",
        f"{RUN}/source/BUNDLE-FORMAT.md",
        f"{RUN}/source/IMPLEMENTATION-NOTES.md",
        f"{RUN}/source/bundle.py",
        f"{RUN}/source/contract.py",
        f"{RUN}/source/suite_records.py",
        f"{RUN}/source/make_role_specs.py",
        f"{RUN}/source/pack_source_closure.py",
        f"{RUN}/controller/README.md",
        f"{RUN}/source/tests/test_bundle.py",
        f"{RUN}/source/tests/test_contract.py",
        "benchmark_runs/volta-gemm/controller/pc_controller.py",
        "benchmark_runs/volta-environment/ncu-collection-shim.py",
        NVCC,
        f"{CUDA}/compute-sanitizer/compute-sanitizer",
        f"{SDK}/profilers/12.9/Nsight_Compute/ncu",
        f"{SDK}/profilers/12.9/Nsight_Compute/target/linux-desktop-glibc_2_11_3-x64/ncu",
        f"{CUBLAS}/lib/libcublas.so.12.9.1.4",
        f"{CUBLAS}/lib/libcublasLt.so.12.9.1.4",
    ]
    for path in sorted((ROOT / f"{RUN}/cases/fixture_bundle").rglob("*")):
        if path.is_file():
            paths.append(path.relative_to(ROOT).as_posix())
    for role_name, *_ in roles():
        paths.append(f"{RUN}/controller/role-{role_name}.spec.json")
    return paths


def build_argv() -> list[str]:
    return [
        NVCC, "-O2", "-std=c++17", "-lineinfo", "-arch=sm_70",
        f"-I{ROOT / ART / 'variants/include'}",
        f"-I{CUBLAS}/include",
        str(ROOT / ART / "variants/kwv07_runner.cu"),
        str(ROOT / ART / "variants/grouped_lowbit.cu"),
        f"-L{CUBLAS}/lib", "-lcublas", "-o", str(ROOT / RUNNER),
    ]


def spec_for(family: str, role_name: str, impl: str, recipe: str,
             mode: str, case: str | None, first_build: bool) -> dict[str, object]:
    suite = "grouped" if family == "grouped" else "w4a16"
    contract = "kwv07-grouped-gated" if family == "grouped" else "kwv07-w4a16"
    app = [RUNNER, "--bundle", BUNDLE, "--suite", suite,
           "--impl", impl, "--mode", mode]
    if case:
        app.extend(["--case", case])
    value: dict[str, object] = {
        "schema_version": 1,
        "project_root": str(ROOT),
        "command_cwd": str(ROOT),
        "paths": [ART, RUN],
        "inputs": declared_inputs(),
        "binary_paths": [RUNNER],
        "contract_id": contract,
        "toolchain": {"root": f"{CUDA}", "require_sanitizer": True},
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
        "role": role_name,
        "timeout": 7200 if recipe == "ncu" else 3600,
        "recipe": recipe,
        "argv": app,
    }
    if first_build:
        value["benchmark"] = {"build_argv": build_argv()}
    return value


def roles() -> list[tuple[str, str, str, str | None, bool]]:
    result = []
    for family, baseline, candidate, profile_case in (
        ("grouped", "split-gated", "paired-gated", "weighted_skewed"),
        ("w4a16", "predecode-w4", "direct-w4", "weighted_low_m"),
    ):
        prefix = "grouped" if family == "grouped" else "w4a16"
        result.extend([
            (f"{prefix}_correctness_baseline", baseline, "baseline", None,
             family == "grouped"),
            (f"{prefix}_correctness_candidate", candidate, "baseline", None, False),
            (f"{prefix}_sanitizer_candidate", candidate, "compute-sanitizer", None, False),
            (f"{prefix}_baseline_timing", baseline, "baseline", None, False),
            (f"{prefix}_candidate_timing", candidate, "baseline", None, False),
            (f"{prefix}_profile_candidate", candidate, "ncu", profile_case, False),
        ])
    return result


def main() -> None:
    controller = ROOT / RUN / "controller"
    controller.mkdir(parents=True, exist_ok=True)
    (ROOT / RUN / "bin/v1").mkdir(parents=True, exist_ok=True)
    for role_name, implementation, recipe, case, first_build in roles():
        family = "grouped" if role_name.startswith("grouped_") else "w4a16"
        value = spec_for(family, role_name, implementation, recipe,
                         "time" if role_name.endswith("_timing") else "check",
                         case, first_build)
        path = controller / f"role-{role_name}.spec.json"
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
