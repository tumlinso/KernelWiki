#!/usr/bin/env python3
"""Run an admitted CUDA controller command with the qualified host tools.

Invoke through Project Control's Python runtime, for example:

  pc-dev python benchmark_runs/volta-rows/controller/pc_controller.py run \
    --spec benchmark_runs/volta-rows/controller/role-rows-sanitizer_candidate.spec.json \
    --json

The spec path may be repository-relative or absolute. This launcher keeps the
controller, toolkit, sanitizer, and Nsight Compute selection consistent across
the GEMM roles; it does not provide a separate execution or admission path.
"""

from __future__ import annotations

import os
from pathlib import Path
import json
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS_ROOT = "/home/tumlinson/.agents/skills"
CODING_WORKFLOW_ROOT = "/home/tumlinson/skills_dev/integrations/coding-workflow-mcp"
CUDA_SCRIPT_ROOT = f"{SKILLS_ROOT}/cuda/scripts"
TOOLKIT_ROOT = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9"
SANITIZER = f"{TOOLKIT_ROOT}/compute-sanitizer/compute-sanitizer"
NCU = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/profilers/12.9/Nsight_Compute/ncu"
NCU_SHIM = REPO_ROOT / "benchmark_runs/volta-rows/controller/ncu-collection-shim.py"


def main() -> None:
    args = list(sys.argv[1:])
    if not args:
        raise SystemExit("usage: pc_controller.py <controller command and arguments>")

    shim_log: str | None = None
    if "--ncu-shim-log" in args:
        index = args.index("--ncu-shim-log")
        if index + 1 >= len(args):
            raise SystemExit("--ncu-shim-log requires an output path")
        shim_log = args[index + 1]
        del args[index:index + 2]

    # A relative --spec is interpreted from this checkout, regardless of the
    # caller's current directory. The controller still enforces each spec's
    # own project_root and admission requirements.
    spec_data: dict[str, object] | None = None
    for index, arg in enumerate(args[:-1]):
        if arg == "--spec" and args[index + 1] != "-":
            spec = Path(args[index + 1])
            if not spec.is_absolute():
                spec = (REPO_ROOT / spec).resolve()
                args[index + 1] = str(spec)
            if spec.is_file():
                spec_data = json.loads(spec.read_text(encoding="utf-8"))

    if args[0] == "run" and isinstance(spec_data, dict) and spec_data.get("recipe") == "ncu":
        if not shim_log:
            raise SystemExit("NCU runs require --ncu-shim-log so the no-clock capture is archived")
        log_path = Path(shim_log)
        if not log_path.is_absolute():
            log_path = REPO_ROOT / log_path
        log_path = log_path.resolve()
        if not log_path.is_relative_to(REPO_ROOT):
            raise SystemExit("--ncu-shim-log must stay within this repository")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["NCU_SHIM_LOG"] = str(log_path)

    os.chdir(REPO_ROOT)
    os.environ["PROJECT_CONTROL_SKILLS_ROOT"] = SKILLS_ROOT
    os.environ["COMPUTE_SANITIZER_BIN"] = SANITIZER
    os.environ["NCU_BIN"] = str(NCU_SHIM)
    os.environ["NCU_REAL_BIN"] = NCU
    os.environ["PATH"] = os.pathsep.join(
        [str(Path(NCU).parent), str(Path(SANITIZER).parent), os.environ.get("PATH", "")]
    )
    pythonpath = [CODING_WORKFLOW_ROOT, CUDA_SCRIPT_ROOT]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    sys.path[:0] = [CODING_WORKFLOW_ROOT, CUDA_SCRIPT_ROOT]
    sys.argv = ["cuda_controller.py", *args]
    runpy.run_path(f"{CUDA_SCRIPT_ROOT}/cuda_controller.py", run_name="__main__")


if __name__ == "__main__":
    main()
