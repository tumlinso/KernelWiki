#!/usr/bin/env python3
"""Verify frozen ordered-frontend and shared GEMM inputs before controller work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-scratch-fallback", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "KERNELWIKI-VOLTA-ORDERED-SOURCE-CLOSURE/1":
        raise SystemExit("wrong source closure format")
    failures=[]
    for entry in manifest.get("files", []):
        repo_path = Path(entry["path"])
        path = repo_path if repo_path.is_absolute() else root / repo_path
        if not path.is_file() and args.allow_scratch_fallback and entry.get("scratch_path"):
            path = Path(entry["scratch_path"])
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        actual=digest(path)
        if actual != entry["sha256"]:
            failures.append(f"sha256 mismatch: {entry['path']}: {actual}")
    if failures:
        raise SystemExit("source closure failed:\n" + "\n".join(failures))
    print(json.dumps({"format":"KERNELWIKI-VOLTA-ORDERED-SOURCE-CHECK/1","files_checked":len(manifest['files']),"valid":True},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
