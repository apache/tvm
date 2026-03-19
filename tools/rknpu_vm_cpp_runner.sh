#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bundle_dir> [runner args...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="$1"
RUNNER_SRC="$ROOT_DIR/tools/rknpu_vm_cpp_runner.cc"
RUNNER_BIN="$ROOT_DIR/build/rknpu_vm_cpp_runner"

g++ -std=c++17 -O2 \
  -I"$ROOT_DIR/include" \
  -I"$ROOT_DIR/3rdparty/tvm-ffi/include" \
  -I"$ROOT_DIR/3rdparty/tvm-ffi/3rdparty/dlpack/include" \
  "$RUNNER_SRC" \
  -L"$ROOT_DIR/build" \
  -L"$ROOT_DIR/build/lib" \
  -ltvm \
  -ltvm_ffi \
  -Wl,-rpath,"$ROOT_DIR/build" \
  -Wl,-rpath,"$ROOT_DIR/build/lib" \
  -o "$RUNNER_BIN"

export LD_LIBRARY_PATH="$ROOT_DIR/build:$ROOT_DIR/build/lib:${LD_LIBRARY_PATH:-}"
export TVM_LIBRARY_PATH="${TVM_LIBRARY_PATH:-$ROOT_DIR/build}"
export TVM_RKNPU_BRIDGE_REAL_SUBMIT="${TVM_RKNPU_BRIDGE_REAL_SUBMIT:-1}"
export TVM_RKNPU_BRIDGE_USE_RELOCS="${TVM_RKNPU_BRIDGE_USE_RELOCS:-1}"
export TVM_RKNPU_BRIDGE_FAIL_ON_FALLBACK="${TVM_RKNPU_BRIDGE_FAIL_ON_FALLBACK:-1}"

"$RUNNER_BIN" "$@"
