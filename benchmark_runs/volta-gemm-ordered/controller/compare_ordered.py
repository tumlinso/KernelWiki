#!/usr/bin/env python3
"""Use the existing six-route correctness matrix with the ordered ELF."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("KERNELWIKI_ROOT", "/home/tumlinson/KernelWiki")).resolve()
MATRIX = ROOT / "benchmark_runs/volta-gemm/supplemental/compare_matrix.py"
RUNNER = ROOT / "benchmark_runs/volta-gemm-ordered/evidence/shared/bin/gemm_runner-source-9ab1ba79270f128d5b3cbbd56da8db991799326604ef87973085981672dd3e9c.bin"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("large_dense", "small_m", "ragged"), required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("kwv04_compare_matrix", MATRIX)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing compare matrix at {MATRIX}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUNNER = RUNNER
    sys.argv = [str(MATRIX), "--case", args.case, "--phase", "correctness"]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
