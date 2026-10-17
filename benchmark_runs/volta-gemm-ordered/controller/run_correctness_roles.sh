#!/usr/bin/env bash
set -euo pipefail

ROOT=${REPO_ROOT:-/home/tumlinson/KernelWiki}
PC_DEV=${PC_DEV:-/home/tumlinson/project-control/scripts/pc-dev}
CONTROLLER="$ROOT/benchmark_runs/volta-gemm/controller/pc_controller.py"
ROLE_SPECS="$ROOT/benchmark_runs/volta-gemm-ordered/controller"
ROLE_OUTPUT="$ROOT/benchmark_runs/volta-gemm-ordered/evidence/attempts"
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID must contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
ATTEMPT="$ROLE_OUTPUT/$RUN_ID"
if [[ -e "$ATTEMPT" ]]; then
  echo "attempt output already exists; choose a new RUN_ID to preserve prior records" >&2
  exit 2
fi
mkdir -p -- "$ATTEMPT"
cd -- "$ROOT"
python3 "$ROLE_SPECS/verify_source_closure.py" \
  --root "$ROOT" --manifest "$ROLE_SPECS/source-closure.json"

roles=(large_dense small_m ragged onehot)
for role in "${roles[@]}"; do
  role_dir="$ATTEMPT/$role"
  mkdir -p -- "$role_dir"
  spec="$ROLE_SPECS/role-$role-correctness.spec.json"
  controller_json="$role_dir/controller.json"
  controller_err="$role_dir/controller.stderr.txt"
  if ! "$PC_DEV" python "$CONTROLLER" run --spec "$spec" --json \
      >"$controller_json" 2>"$controller_err"; then
    status=1
  else
    status=0
  fi
  "$PC_DEV" python "$ROLE_SPECS/export_receipt.py" \
    --root "$ROOT" --execution "$controller_json" --out-dir "$role_dir" \
    --role "$role" --spec "$spec"
  python3 - "$controller_json" "$role" <<'PY'
import json, sys
from pathlib import Path
result=json.loads(Path(sys.argv[1]).read_text())
if result.get('ok') is not True or result.get('status') != 'succeeded' or result.get('returncode') != 0:
    raise SystemExit(f"controller role {sys.argv[2]} did not succeed; raw result preserved")
PY
  if [[ "$status" -ne 0 ]]; then
    echo "controller launcher failed for $role; raw output and receipt preserved at $role_dir" >&2
    exit 1
  fi
  echo "completed role $role; receipt preserved at $role_dir"
done
