#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/tumlinson/KernelWiki}
FRONTEND_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FROZEN_VARIANTS="$REPO_ROOT/artifacts/kernels/volta-gemm/variants"
HPC_SDK=${HPC_SDK:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.1}
CUDA_ROOT=${CUDA_ROOT:-$HPC_SDK/cuda/12.9}
MATH_ROOT=${MATH_ROOT:-$HPC_SDK/math_libs/12.9}
CUTLASS_ROOT=${CUTLASS_ROOT:-/tmp/kernelwiki-gemm-prep.X4Sd0K/cutlass-v2.11.0}
CUBLAS_LIB="$MATH_ROOT/targets/x86_64-linux/lib"
CUBLAS_INCLUDE="$MATH_ROOT/targets/x86_64-linux/include"
OUTPUT_BIN=${OUTPUT_BIN:-$REPO_ROOT/benchmark_runs/volta-gemm-ordered/bin/gemm_runner}
ARCHIVE_DIR="$REPO_ROOT/benchmark_runs/volta-gemm-ordered/evidence/shared/bin"
BUILD_DIR="$REPO_ROOT/benchmark_runs/volta-gemm-ordered/evidence/shared"
SOURCE_SHA=$(sha256sum "$FRONTEND_DIR/gemm_runner.cu" | awk '{print $1}')
ARCHIVE_BIN="$ARCHIVE_DIR/gemm_runner-source-$SOURCE_SHA.bin"
BUILD_MANIFEST="$BUILD_DIR/runner-build-source-$SOURCE_SHA.json"
mkdir -p -- "$(dirname -- "$OUTPUT_BIN")" "$ARCHIVE_DIR" "$BUILD_DIR"

if [[ -e "$ARCHIVE_BIN" || -e "$BUILD_MANIFEST" ]]; then
  [[ -f "$ARCHIVE_BIN" && -f "$BUILD_MANIFEST" ]] || {
    echo "incomplete prior runner archive; refusing to rebuild" >&2
    exit 2
  }
  python3 - "$BUILD_MANIFEST" "$ARCHIVE_BIN" "$SOURCE_SHA" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text())
binary=Path(sys.argv[2])
actual=hashlib.sha256(binary.read_bytes()).hexdigest()
if manifest.get('format') != 'KERNELWIKI-VOLTA-ORDERED-GEMM-BUILD/1': raise SystemExit('wrong build manifest format')
if manifest.get('source_sha256') != sys.argv[3]: raise SystemExit('archived runner source hash differs')
if manifest.get('binary_sha256') != actual: raise SystemExit('archived runner binary hash differs')
if manifest.get('archive_sha256') != actual: raise SystemExit('archived runner archive hash differs')
PY
  cp -- "$ARCHIVE_BIN" "$OUTPUT_BIN"
else
  "$CUDA_ROOT/bin/nvcc" \
    -std=c++17 -O3 -lineinfo -arch=sm_70 \
    -I"$FROZEN_VARIANTS" -I"$CUBLAS_INCLUDE" -I"$CUTLASS_ROOT/include" \
    "$FRONTEND_DIR/gemm_runner.cu" \
    "$FROZEN_VARIANTS/gemm_baselines.cu" \
    "$FROZEN_VARIANTS/volta_ptx_gemm.cu" \
    "$FROZEN_VARIANTS/gemm_e09_smoke.cu" \
    -L"$CUBLAS_LIB" -Xlinker -rpath -Xlinker "$CUBLAS_LIB" -lcublas \
    -o "$OUTPUT_BIN"
  BINARY_SHA=$(sha256sum "$OUTPUT_BIN" | awk '{print $1}')
  cp -- "$OUTPUT_BIN" "$ARCHIVE_BIN"
  cmp -s -- "$OUTPUT_BIN" "$ARCHIVE_BIN"
  python3 - "$BUILD_MANIFEST" "$FRONTEND_DIR/gemm_runner.cu" "$OUTPUT_BIN" "$ARCHIVE_BIN" "$SOURCE_SHA" "$BINARY_SHA" <<'PY'
import json, sys
from pathlib import Path
out, source, binary, archive, source_sha, binary_sha = sys.argv[1:]
record={
    'format':'KERNELWIKI-VOLTA-ORDERED-GEMM-BUILD/1',
    'source':source,
    'source_sha256':source_sha,
    'binary':binary,
    'binary_sha256':binary_sha,
    'archive':archive,
    'archive_sha256':binary_sha,
}
Path(out).write_text(json.dumps(record,indent=2)+'\n')
PY
fi
cmp -s -- "$OUTPUT_BIN" "$ARCHIVE_BIN"
cat -- "$BUILD_MANIFEST"
