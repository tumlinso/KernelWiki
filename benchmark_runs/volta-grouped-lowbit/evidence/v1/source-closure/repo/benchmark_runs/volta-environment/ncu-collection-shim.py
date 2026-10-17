#!/usr/bin/env python3
"""Inject the no-clock-change NCU collection policy and record exact argv."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flag_values(args: list[str], flag: str) -> list[str]:
    values = []
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            values.append(args[index + 1])
        elif arg == flag:
            values.append("")
        elif arg.startswith(flag + "="):
            values.append(arg.split("=", 1)[1])
    return values


def main() -> int:
    actual = os.environ.get("NCU_REAL_BIN")
    if not actual:
        print("NCU_REAL_BIN must name the actual Nsight Compute executable", file=sys.stderr)
        return 64
    actual_path = Path(actual).resolve()
    if not actual_path.is_file() or not os.access(actual_path, os.X_OK):
        print(f"actual NCU is unavailable: {actual_path}", file=sys.stderr)
        return 64
    original = sys.argv[1:]
    passthrough = any(
        arg in {"--help", "-h", "--version", "--import"} or
        arg.startswith("--import=") or
        arg.startswith(("--list-", "--query-"))
        for arg in original[:2]
    )
    effective = list(original)
    if not passthrough:
        clocks = flag_values(effective, "--clock-control")
        sources = flag_values(effective, "--import-source")
        if len(clocks) > 1 or len(sources) > 1:
            print("duplicate --clock-control or --import-source flags are disallowed", file=sys.stderr)
            return 64
        clock = clocks[0] if clocks else None
        source = sources[0] if sources else None
        if clock not in (None, "none"):
            print("clock changes are disallowed; --clock-control must be none", file=sys.stderr)
            return 64
        if source not in (None, "yes"):
            print("source import is required; --import-source must be yes", file=sys.stderr)
            return 64
        if clock is None:
            effective[0:0] = ["--clock-control", "none"]
        if source is None:
            effective[0:0] = ["--import-source", "yes"]
    record_path = os.environ.get("NCU_SHIM_LOG")
    if record_path:
        shim_path = Path(__file__).resolve()
        record = {
            "policy": "passthrough" if passthrough else "collection-clock-control-none-source-import-yes",
            "shim_path": str(shim_path), "shim_sha256": sha256(shim_path),
            "actual_ncu_path": str(actual_path), "actual_ncu_sha256": sha256(actual_path),
            "argv_received": original, "argv_executed": effective,
        }
        log = Path(record_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    os.execv(str(actual_path), [str(actual_path), *effective])
    return 127


if __name__ == "__main__":
    sys.exit(main())
