#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TVM_PYTHON="${TVM_PYTHON:-python}"
TVM_RAW_DIST="${TVM_RAW_DIST:-${REPO_ROOT}/dist/tvm-raw}"
TVM_INJECTED_DIST="${TVM_INJECTED_DIST:-${REPO_ROOT}/dist/tvm-injected}"
TVM_WHEELHOUSE="${TVM_WHEELHOUSE:-${REPO_ROOT}/wheelhouse}"
TVM_CUDA_BUILD_DIR="${TVM_CUDA_BUILD_DIR:-${REPO_ROOT}/build-wheel-cuda}"
TVM_BASE_BUILD_DIR="${TVM_BASE_BUILD_DIR:-${REPO_ROOT}/build-wheel-base}"
TVM_USE_LLVM="${TVM_USE_LLVM:-ON}"
TVM_USE_CUDA="${TVM_USE_CUDA:-ON}"
TVM_CUDA_ARCHITECTURES="${TVM_CUDA_ARCHITECTURES:-75}"
TVM_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
TVM_WHEEL_DIST_NAME="${TVM_WHEEL_DIST_NAME:-}"
TVM_WHEEL_DIST_VERSION="${TVM_WHEEL_DIST_VERSION:-}"
TVM_SKIP_CUDA="${TVM_SKIP_CUDA:-0}"
TVM_SKIP_REPAIR="${TVM_SKIP_REPAIR:-0}"
TVM_BUILD_NO_ISOLATION="${TVM_BUILD_NO_ISOLATION:-0}"
TVM_KEEP_BUILD_DIRS="${TVM_KEEP_BUILD_DIRS:-0}"

usage() {
  cat <<'EOF'
Usage: ci/scripts/package/build_tvm_wheel.sh [all|cuda|wheel|inject|repair|verify|upload|verify-pypi]

Environment knobs:
  TVM_USE_LLVM                 LLVM config for the base wheel, default ON
  TVM_USE_CUDA                 CUDA root or ON for the sidecar build, default ON
  TVM_CUDA_ARCHITECTURES       CMake CUDA arch list, default 75
  TVM_WHEEL_DIST_NAME          Optional distribution rename for TestPyPI
  TVM_WHEEL_DIST_VERSION       Optional distribution version rewrite
  TVM_UPLOAD_REPOSITORY_URL    Twine repository URL, e.g. TestPyPI legacy URL
  TVM_SKIP_CUDA=1              Do not build/inject libtvm_runtime_cuda.so
  TVM_SKIP_REPAIR=1            Keep injected wheel as final wheel
  TVM_BUILD_NO_ISOLATION=1     Pass --no-isolation to python -m build
  TVM_KEEP_BUILD_DIRS=1        Reuse CMake build dirs instead of cleaning them
  TVM_TEST_INDEX_URL           Package index for verify-pypi, default TestPyPI
  TVM_EXTRA_INDEX_URL          Extra package index for dependencies, default PyPI
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    return 1
  fi
}

require_pypa_build() {
  local check_dir
  check_dir="$(mktemp -d)"
  if ! (cd "$check_dir" && "$TVM_PYTHON" -m build --version >/dev/null 2>&1); then
    rm -rf "$check_dir"
    echo "error: PyPA build is missing; install it with: ${TVM_PYTHON} -m pip install build" >&2
    return 1
  fi
  rm -rf "$check_dir"
}

single_wheel() {
  local dir="$1"
  mapfile -t wheels < <(find "$dir" -maxdepth 1 -type f -name '*.whl' | sort)
  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "error: expected exactly one wheel under ${dir}, found ${#wheels[@]}" >&2
    printf '%s\n' "${wheels[@]}" >&2
    return 1
  fi
  echo "${wheels[0]}"
}

wheel_metadata_field() {
  local wheel="$1"
  local field="$2"
  "$TVM_PYTHON" - "$wheel" "$field" <<'PY'
from email.parser import Parser
from pathlib import Path
import sys
import zipfile

wheel = Path(sys.argv[1])
field = sys.argv[2]
with zipfile.ZipFile(wheel) as zf:
    metadata_name = next(name for name in zf.namelist() if name.endswith(".dist-info/METADATA"))
    metadata = Parser().parsestr(zf.read(metadata_name).decode("utf-8"))
print(metadata[field])
PY
}

cuda_runtime_path() {
  if [[ ! -d "$TVM_CUDA_BUILD_DIR" ]]; then
    return 0
  fi
  find "$TVM_CUDA_BUILD_DIR" -type f -name 'libtvm_runtime_cuda.so' | sort | tail -n 1
}

build_cuda_runtime() {
  if [[ "$TVM_SKIP_CUDA" == "1" ]]; then
    echo "Skipping CUDA sidecar build because TVM_SKIP_CUDA=1"
    return 0
  fi

  require_cmd cmake
  echo "Building libtvm_runtime_cuda.so in ${TVM_CUDA_BUILD_DIR}"
  if [[ "$TVM_KEEP_BUILD_DIRS" != "1" ]]; then
    rm -rf "$TVM_CUDA_BUILD_DIR"
  fi
  local cuda_compiler_args=()
  if [[ -x "${TVM_USE_CUDA}/bin/nvcc" ]]; then
    cuda_compiler_args+=(-DCMAKE_CUDA_COMPILER="${TVM_USE_CUDA}/bin/nvcc")
  fi
  cmake -S "$REPO_ROOT" -B "$TVM_CUDA_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DTVM_BUILD_PYTHON_MODULE=ON \
    -DUSE_CUDA="$TVM_USE_CUDA" \
    -DUSE_LLVM=OFF \
    -DUSE_CUBLAS=OFF \
    -DUSE_CUDNN=OFF \
    -DUSE_CUTLASS=OFF \
    -DUSE_NCCL=OFF \
    -DUSE_NVTX=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$TVM_CUDA_ARCHITECTURES" \
    "${cuda_compiler_args[@]}"

  cmake --build "$TVM_CUDA_BUILD_DIR" --target tvm_runtime tvm_runtime_cuda --parallel "$TVM_BUILD_PARALLEL_LEVEL"

  local cuda_lib
  cuda_lib="$(cuda_runtime_path)"
  if [[ -z "$cuda_lib" ]]; then
    echo "error: libtvm_runtime_cuda.so was not produced" >&2
    return 1
  fi
  if [[ "$(uname -s)" == "Linux" ]]; then
    require_cmd patchelf
    patchelf --set-rpath '$ORIGIN' "$cuda_lib"
  fi
  echo "CUDA sidecar: ${cuda_lib}"
}

build_base_wheel() {
  require_pypa_build
  rm -rf "$TVM_RAW_DIST"
  mkdir -p "$TVM_RAW_DIST"
  if [[ "$TVM_KEEP_BUILD_DIRS" != "1" ]]; then
    rm -rf "$TVM_BASE_BUILD_DIR"
  fi

  local build_flags=()
  if [[ "$TVM_BUILD_NO_ISOLATION" == "1" ]]; then
    build_flags+=(--no-isolation)
  fi

  echo "Building base TVM wheel with LLVM=${TVM_USE_LLVM}, CUDA=OFF"
  (
    cd "$TVM_RAW_DIST"
    CMAKE_ARGS="-DUSE_LLVM=${TVM_USE_LLVM} -DUSE_CUDA=OFF -DBUILD_TESTING=OFF -DTVM_BUILD_PYTHON_MODULE=ON ${TVM_EXTRA_CMAKE_ARGS:-}" \
      "$TVM_PYTHON" -m build --wheel --outdir "$TVM_RAW_DIST" \
        "${build_flags[@]}" \
        -Cbuild-dir="$TVM_BASE_BUILD_DIR" \
        "$REPO_ROOT"
  )

  single_wheel "$TVM_RAW_DIST" >/dev/null
}

inject_cuda_runtime() {
  rm -rf "$TVM_INJECTED_DIST"
  mkdir -p "$TVM_INJECTED_DIST"

  local raw_wheel
  raw_wheel="$(single_wheel "$TVM_RAW_DIST")"

  local inject_args=(--output-dir "$TVM_INJECTED_DIST")
  if [[ "$TVM_SKIP_CUDA" != "1" ]]; then
    local cuda_lib
    cuda_lib="$(cuda_runtime_path)"
    if [[ -z "$cuda_lib" ]]; then
      echo "error: CUDA sidecar missing; run the 'cuda' step first" >&2
      return 1
    fi
    inject_args+=(--cuda-runtime "$cuda_lib")
  fi
  if [[ -n "$TVM_WHEEL_DIST_NAME" ]]; then
    inject_args+=(--distribution-name "$TVM_WHEEL_DIST_NAME")
  fi
  if [[ -n "$TVM_WHEEL_DIST_VERSION" ]]; then
    inject_args+=(--distribution-version "$TVM_WHEEL_DIST_VERSION")
  fi

  echo "Injecting sidecar/metadata into ${raw_wheel}"
  "$TVM_PYTHON" "$SCRIPT_DIR/inject_cuda_runtime.py" "$raw_wheel" "${inject_args[@]}"
}

auditwheel_excludes() {
  local cuda_lib="$1"
  local seen=" libtvm_runtime_cuda.so libcuda.so.1 libcuda.so libcudart.so.11.0 libcudart.so.12 libcudart.so.12.0 "

  printf '%s\n' "--exclude" "libtvm_runtime_cuda.so"
  printf '%s\n' "--exclude" "libcuda.so.1"
  printf '%s\n' "--exclude" "libcuda.so"
  printf '%s\n' "--exclude" "libcudart.so.11.0"
  printf '%s\n' "--exclude" "libcudart.so.12"
  printf '%s\n' "--exclude" "libcudart.so.12.0"

  if [[ -n "$cuda_lib" ]] && command -v readelf >/dev/null 2>&1; then
    while IFS= read -r needed; do
      case "$needed" in
        libcuda.so*|libcudart.so*|libnv*.so*)
          if [[ "$seen" != *" ${needed} "* ]]; then
            seen+="${needed} "
            printf '%s\n' "--exclude" "$needed"
          fi
          ;;
      esac
    done < <(readelf -d "$cuda_lib" | sed -n 's/.*Shared library: \[\(.*\)\].*/\1/p')
  fi
}

repair_wheel() {
  rm -rf "$TVM_WHEELHOUSE"
  mkdir -p "$TVM_WHEELHOUSE"

  local injected_wheel
  injected_wheel="$(single_wheel "$TVM_INJECTED_DIST")"

  if [[ "$TVM_SKIP_REPAIR" == "1" ]]; then
    cp "$injected_wheel" "$TVM_WHEELHOUSE/"
    echo "Repair skipped; final wheel copied to ${TVM_WHEELHOUSE}"
    return 0
  fi

  case "$(uname -s)" in
    Linux)
      require_cmd auditwheel
      local cuda_lib
      cuda_lib="$(cuda_runtime_path || true)"
      mapfile -t exclude_args < <(auditwheel_excludes "$cuda_lib")
      echo "Repairing Linux wheel with auditwheel"
      auditwheel repair "${exclude_args[@]}" -w "$TVM_WHEELHOUSE" "$injected_wheel"
      ;;
    Darwin)
      require_cmd delocate-wheel
      echo "Repairing macOS wheel with delocate"
      delocate-wheel --ignore-missing-dependencies -w "$TVM_WHEELHOUSE" -v "$injected_wheel"
      ;;
    *)
      cp "$injected_wheel" "$TVM_WHEELHOUSE/"
      echo "No repair step for this platform; final wheel copied to ${TVM_WHEELHOUSE}"
      ;;
  esac

  single_wheel "$TVM_WHEELHOUSE" >/dev/null
}

verify_wheel() {
  local final_wheel
  final_wheel="$(single_wheel "$TVM_WHEELHOUSE")"

  local venv="${TVM_VERIFY_VENV:-${REPO_ROOT}/build-wheel-verify-venv}"
  rm -rf "$venv"
  "$TVM_PYTHON" -m venv "$venv"

  local venv_python="${venv}/bin/python"
  if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == CYGWIN* ]]; then
    venv_python="${venv}/Scripts/python.exe"
  fi

  "$venv_python" -m pip install --upgrade pip
  "$venv_python" -m pip install --extra-index-url "${TVM_EXTRA_INDEX_URL:-https://pypi.org/simple}" "$final_wheel"
  "$venv_python" - <<'PY'
from pathlib import Path
import tvm

root = Path(tvm.__file__).resolve().parent
print("tvm version:", tvm.__version__)
print("tvm package:", root)
print("llvm enabled:", tvm.runtime.enabled("llvm"))
print("cuda runtime enabled:", tvm.runtime.enabled("cuda"))
assert (root / "lib" / "libtvm_runtime.so").exists()
cuda_sidecar = root / "lib" / "libtvm_runtime_cuda.so"
print("cuda sidecar present:", cuda_sidecar.exists())
PY
}

upload_wheel() {
  require_cmd twine
  local repo_args=()
  if [[ -n "${TVM_UPLOAD_REPOSITORY_URL:-}" ]]; then
    repo_args+=(--repository-url "$TVM_UPLOAD_REPOSITORY_URL")
  fi
  twine upload "${repo_args[@]}" "$TVM_WHEELHOUSE"/*
}

verify_pypi_wheel() {
  local final_wheel
  final_wheel="$(single_wheel "$TVM_WHEELHOUSE")"

  local package_name package_version
  package_name="$(wheel_metadata_field "$final_wheel" Name)"
  package_version="$(wheel_metadata_field "$final_wheel" Version)"

  local index_url="${TVM_TEST_INDEX_URL:-https://test.pypi.org/simple/}"
  local extra_index_url="${TVM_EXTRA_INDEX_URL:-https://pypi.org/simple}"
  local venv="${TVM_VERIFY_PYPI_VENV:-${REPO_ROOT}/build-wheel-verify-pypi-venv}"
  rm -rf "$venv"
  "$TVM_PYTHON" -m venv "$venv"

  local venv_python="${venv}/bin/python"
  if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == CYGWIN* ]]; then
    venv_python="${venv}/Scripts/python.exe"
  fi

  "$venv_python" -m pip install --upgrade pip
  "$venv_python" -m pip install \
    --index-url "$index_url" \
    --extra-index-url "$extra_index_url" \
    "${package_name}==${package_version}"
  "$venv_python" - <<'PY'
from pathlib import Path
import tvm

root = Path(tvm.__file__).resolve().parent
print("tvm version:", tvm.__version__)
print("tvm package:", root)
print("llvm enabled:", tvm.runtime.enabled("llvm"))
print("cuda runtime enabled:", tvm.runtime.enabled("cuda"))
assert (root / "lib" / "libtvm_runtime.so").exists()
print("cuda sidecar present:", (root / "lib" / "libtvm_runtime_cuda.so").exists())
PY
}

main() {
  local step="${1:-all}"
  case "$step" in
    all)
      build_cuda_runtime
      build_base_wheel
      inject_cuda_runtime
      repair_wheel
      verify_wheel
      ;;
    cuda) build_cuda_runtime ;;
    wheel) build_base_wheel ;;
    inject) inject_cuda_runtime ;;
    repair) repair_wheel ;;
    verify) verify_wheel ;;
    upload) upload_wheel ;;
    verify-pypi) verify_pypi_wheel ;;
    -h|--help|help) usage ;;
    *)
      usage >&2
      return 1
      ;;
  esac
}

main "$@"
