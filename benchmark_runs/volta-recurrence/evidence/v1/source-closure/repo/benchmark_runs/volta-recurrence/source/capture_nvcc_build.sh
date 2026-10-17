#!/usr/bin/env bash
set -euo pipefail

if (($# < 4)) || [[ "$3" != "--" ]]; then
  printf 'usage: capture_nvcc_build.sh STDOUT_PATH STDERR_PATH -- NVCC ARG...\n' >&2
  exit 64
fi
stdout_path="$1"
stderr_path="$2"
shift 3
mkdir -p "$(dirname "$stdout_path")" "$(dirname "$stderr_path")"
set +e
"$@" >"$stdout_path" 2>"$stderr_path"
status=$?
set -e
cat "$stdout_path"
cat "$stderr_path" >&2
exit "$status"
