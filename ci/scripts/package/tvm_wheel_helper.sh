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
TVM_WHEELHOUSE="${TVM_WHEELHOUSE:-${REPO_ROOT}/wheelhouse}"
TVM_CUDA_BUILD_DIR="${TVM_CUDA_BUILD_DIR:-${REPO_ROOT}/build-wheel-cuda}"
TVM_USE_LLVM="${TVM_USE_LLVM:-llvm-config --link-static}"
TVM_USE_CUDA="${TVM_USE_CUDA:-ON}"
TVM_CUDA_ARCHITECTURES="${TVM_CUDA_ARCHITECTURES:-75}"
TVM_BUILD_PARALLEL_LEVEL="${TVM_BUILD_PARALLEL_LEVEL:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}}"
TVM_WHEEL_DIST_NAME="${TVM_WHEEL_DIST_NAME:-}"
TVM_WHEEL_DIST_VERSION="${TVM_WHEEL_DIST_VERSION:-}"
TVM_SKIP_CUDA="${TVM_SKIP_CUDA:-0}"
TVM_SKIP_REPAIR="${TVM_SKIP_REPAIR:-0}"
TVM_KEEP_BUILD_DIRS="${TVM_KEEP_BUILD_DIRS:-0}"

usage() {
  cat <<'EOF'
Usage: ci/scripts/package/tvm_wheel_helper.sh [cuda|manylinux-cuda|cibw-repair|validate|verify|verify-installed|upload|verify-pypi]

Environment knobs:
  TVM_USE_LLVM                 LLVM config used by repair helpers, default "llvm-config --link-static"
  TVM_USE_CUDA                 CUDA root or ON for the CUDA build, default ON
  TVM_CUDA_ARCHITECTURES       CMake CUDA arch list, default 75
  TVM_WHEEL_DIST_NAME          Optional distribution rename for TestPyPI
  TVM_WHEEL_DIST_VERSION       Optional distribution version rewrite
  TVM_UPLOAD_REPOSITORY_URL    Twine repository URL, e.g. TestPyPI legacy URL
  TVM_SKIP_CUDA=1              Do not build/inject libtvm_runtime_cuda.so
  TVM_SKIP_REPAIR=1            Keep injected wheel as final wheel
  TVM_KEEP_BUILD_DIRS=1        Reuse CMake build dirs instead of cleaning them
  TVM_MANYLINUX_IMAGE          manylinux image tag for manylinux-cuda
  TVM_ARCH                     Target architecture for manylinux-cuda
  TVM_AUDITWHEEL_PLAT          Optional auditwheel --plat value
  TVM_EXPECT_WHEEL_PLATFORM_TAG
                                Require the final wheel filename to include this tag
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

single_wheel() {
  local dir="$1"
  local wheels=()
  local wheel
  while IFS= read -r wheel; do
    wheels+=("$wheel")
  done < <(find "$dir" -maxdepth 1 -type f -name '*.whl' | sort)
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

run_manylinux_cuda_container() {
  if [[ "$TVM_SKIP_CUDA" == "1" ]]; then
    echo "Skipping manylinux CUDA build because TVM_SKIP_CUDA=1"
    return 0
  fi

  require_cmd docker
  require_cmd curl
  if [[ -z "${TVM_MANYLINUX_IMAGE:-}" ]]; then
    echo "error: TVM_MANYLINUX_IMAGE is required for manylinux-cuda" >&2
    return 1
  fi
  if [[ -z "${TVM_ARCH:-}" ]]; then
    echo "error: TVM_ARCH is required for manylinux-cuda" >&2
    return 1
  fi

  local image="quay.io/pypa/${TVM_MANYLINUX_IMAGE}_${TVM_ARCH}:latest"
  local container="tvm_wheel_cuda_${GITHUB_RUN_ID:-local}_${GITHUB_RUN_ATTEMPT:-1}_${TVM_ARCH}"
  docker pull "$image"
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker run --name "$container" -d \
    --workdir /workspace \
    --volume "${REPO_ROOT}:/workspace" \
    "$image" tail -f /dev/null
  trap "docker rm -f '${container}' >/dev/null 2>&1 || true" EXIT

  local cuda_rpm="cuda-repo-rhel8-13-0-local-13.0.2_580.95.05-1.${TVM_ARCH}.rpm"
  curl -fsSLo "$cuda_rpm" "https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/${cuda_rpm}"
  docker cp "$cuda_rpm" "${container}:/${cuda_rpm}"
  rm "$cuda_rpm"
  docker exec "$container" bash -lc "
    rpm -i /${cuda_rpm} && \
    dnf clean all && \
    dnf -y --disablerepo=epel install cuda-toolkit-13-0 && \
    rm /${cuda_rpm} && \
    dnf clean all"

  docker exec \
    -e TVM_PYTHON=/opt/python/cp310-cp310/bin/python \
    -e TVM_USE_CUDA=/usr/local/cuda \
    -e TVM_CUDA_ARCHITECTURES="$TVM_CUDA_ARCHITECTURES" \
    -e TVM_SKIP_CUDA="$TVM_SKIP_CUDA" \
    -e CMAKE_BUILD_PARALLEL_LEVEL="$TVM_BUILD_PARALLEL_LEVEL" \
    -e TVM_BUILD_PARALLEL_LEVEL="$TVM_BUILD_PARALLEL_LEVEL" \
    "$container" bash -lc '
      set -eux
      export PATH=/opt/python/cp310-cp310/bin:/usr/local/cuda/bin:$PATH
      python -m pip install -U pip cmake ninja
      python --version
      cmake --version
      nvcc --version
      ci/scripts/package/tvm_wheel_helper.sh cuda'

  docker exec "$container" bash -lc \
    "chown -R $(id -u):$(id -g) /workspace/build-wheel-cuda || true"
}

build_cuda_runtime() {
  if [[ "$TVM_SKIP_CUDA" == "1" ]]; then
    echo "Skipping CUDA build because TVM_SKIP_CUDA=1"
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
  echo "CUDA runtime: ${cuda_lib}"
}

inject_wheel_file() {
  local raw_wheel="$1"
  local output_dir="$2"
  rm -rf "$output_dir"
  mkdir -p "$output_dir"

  local inject_args=(--output-dir "$output_dir")
  if [[ "$TVM_SKIP_CUDA" != "1" ]]; then
    local cuda_lib
    cuda_lib="$(cuda_runtime_path)"
    if [[ -z "$cuda_lib" ]]; then
      echo "error: CUDA runtime missing; run the 'cuda' step first" >&2
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
  if [[ "$(uname -s)" == "Linux" ]]; then
    inject_args+=(--set-rpath '$ORIGIN')
  fi

  echo "Injecting CUDA runtime/metadata into ${raw_wheel}"
  "$TVM_PYTHON" "$SCRIPT_DIR/inject_cuda_runtime.py" "$raw_wheel" "${inject_args[@]}"
}

auditwheel_excludes() {
  local cuda_lib="$1"
  local seen
  seen=" libtvm_ffi.so "
  seen+="libcuda.so.1 libcuda.so libcudart.so.11.0 libcudart.so.12 libcudart.so.12.0 "

  printf '%s\n' "--exclude" "libtvm_ffi.so"
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

llvm_libdir() {
  if [[ "$TVM_USE_LLVM" == "OFF" || "$TVM_USE_LLVM" == "0" ]]; then
    return 0
  fi
  local -a llvm_config
  read -r -a llvm_config <<<"$TVM_USE_LLVM"
  if [[ "${#llvm_config[@]}" -eq 0 ]]; then
    return 0
  fi
  if command -v "${llvm_config[0]}" >/dev/null 2>&1; then
    "${llvm_config[@]}" --libdir
  elif [[ -x "${llvm_config[0]}" ]]; then
    "${llvm_config[@]}" --libdir
  fi
}

llvm_prefix() {
  if [[ "$TVM_USE_LLVM" == "OFF" || "$TVM_USE_LLVM" == "0" ]]; then
    return 0
  fi
  local -a llvm_config
  read -r -a llvm_config <<<"$TVM_USE_LLVM"
  if [[ "${#llvm_config[@]}" -eq 0 ]]; then
    return 0
  fi
  if command -v "${llvm_config[0]}" >/dev/null 2>&1 || [[ -x "${llvm_config[0]}" ]]; then
    "${llvm_config[@]}" --prefix
  fi
}

repair_wheel_to_dir() {
  local injected_wheel="$1"
  local output_dir="$2"
  mkdir -p "$output_dir"
  if [[ "$TVM_SKIP_REPAIR" == "1" ]]; then
    cp "$injected_wheel" "$output_dir/"
    echo "Repair skipped; final wheel copied to ${output_dir}"
    return 0
  fi

  case "$(uname -s)" in
    Linux)
      require_cmd auditwheel
      local cuda_lib
      cuda_lib="$(cuda_runtime_path || true)"
      local exclude_args=()
      local exclude_arg
      while IFS= read -r exclude_arg; do
        exclude_args+=("$exclude_arg")
      done < <(auditwheel_excludes "$cuda_lib")
      echo "Repairing Linux wheel with auditwheel"
      (
        auditwheel_libdir=""
        trap '[[ -z "${auditwheel_libdir:-}" ]] || rm -rf "$auditwheel_libdir"' EXIT
        auditwheel_plat_args=()
        if [[ -n "${TVM_AUDITWHEEL_PLAT:-}" ]]; then
          auditwheel_plat_args+=(--plat "$TVM_AUDITWHEEL_PLAT")
        fi
        llvm_dir="$(llvm_libdir || true)"
        if [[ -n "${llvm_dir:-}" && -d "$llvm_dir" ]]; then
          auditwheel_libdir="$(mktemp -d)"
          shopt -s nullglob
          lib=""
          for lib in "$llvm_dir"/*.so "$llvm_dir"/*.so.*; do
            case "$(basename "$lib")" in
              libstdc++*|libgcc*|libgomp*|libatomic*|libasan*|libtsan*|libubsan*)
                ;;
              *)
                ln -sf "$lib" "$auditwheel_libdir/$(basename "$lib")"
                ;;
            esac
          done
          shopt -u nullglob
          echo "Adding filtered LLVM libdir to LD_LIBRARY_PATH for auditwheel: ${auditwheel_libdir}"
          export LD_LIBRARY_PATH="${auditwheel_libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        fi
        auditwheel repair "${auditwheel_plat_args[@]}" "${exclude_args[@]}" \
          -w "$output_dir" "$injected_wheel"
      )
      ;;
    Darwin)
      require_cmd delocate-wheel
      echo "Repairing macOS wheel with delocate"
      (
        llvm_dir="$(llvm_libdir || true)"
        if [[ -n "${llvm_dir:-}" && -d "$llvm_dir" ]]; then
          echo "Adding LLVM libdir to DYLD_LIBRARY_PATH for delocate: ${llvm_dir}"
          export DYLD_LIBRARY_PATH="${llvm_dir}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
        fi
        delocate-wheel \
          --ignore-missing-dependencies \
          --exclude libtvm_ffi.dylib \
          -w "$output_dir" \
          -v "$injected_wheel"
      )
      ;;
    *)
      cp "$injected_wheel" "$output_dir/"
      echo "No repair step for this platform; final wheel copied to ${output_dir}"
      ;;
  esac

  single_wheel "$output_dir" >/dev/null
}

cibw_repair_wheel() {
  local raw_wheel="$1"
  local dest_dir="$2"
  local injected_dir
  injected_dir="$(mktemp -d)"
  inject_wheel_file "$raw_wheel" "$injected_dir"
  local injected_wheel
  injected_wheel="$(single_wheel "$injected_dir")"
  repair_wheel_to_dir "$injected_wheel" "$dest_dir"
  rm -rf "$injected_dir"
}

validate_wheel_elf() {
  local final_wheel
  final_wheel="$(single_wheel "$TVM_WHEELHOUSE")"
  if [[ "$(uname -s)" == "Linux" ]]; then
    "$TVM_PYTHON" "$SCRIPT_DIR/validate_wheel_elf.py" "$final_wheel"
  fi
}

verify_wheel() {
  local final_wheel
  final_wheel="$(single_wheel "$TVM_WHEELHOUSE")"
  if [[ -n "${TVM_EXPECT_WHEEL_PLATFORM_TAG:-}" ]]; then
    if [[ "$(basename "$final_wheel")" != *"${TVM_EXPECT_WHEEL_PLATFORM_TAG}"* ]]; then
      echo "error: expected final wheel tag ${TVM_EXPECT_WHEEL_PLATFORM_TAG}, got ${final_wheel}" >&2
      return 1
    fi
  fi
  validate_wheel_elf

  local venv="${TVM_VERIFY_VENV:-${REPO_ROOT}/build-wheel-verify-venv}"
  rm -rf "$venv"
  "$TVM_PYTHON" -m venv "$venv"

  local venv_python="${venv}/bin/python"
  if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == CYGWIN* ]]; then
    venv_python="${venv}/Scripts/python.exe"
  fi

  "$venv_python" -m pip install --upgrade pip
  "$venv_python" -m pip install --extra-index-url "${TVM_EXTRA_INDEX_URL:-https://pypi.org/simple}" "$final_wheel"
  "$venv_python" "$SCRIPT_DIR/verify_tvm_install.py"
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
  "$venv_python" "$SCRIPT_DIR/verify_tvm_install.py"
}

main() {
  local step="${1:-help}"
  case "$step" in
    cuda) build_cuda_runtime ;;
    manylinux-cuda) run_manylinux_cuda_container ;;
    cibw-repair)
      if [[ "$#" -ne 3 ]]; then
        echo "error: cibw-repair requires <wheel> <dest-dir>" >&2
        return 1
      fi
      cibw_repair_wheel "$2" "$3"
      ;;
    validate) validate_wheel_elf ;;
    verify) verify_wheel ;;
    verify-installed) "$TVM_PYTHON" "$SCRIPT_DIR/verify_tvm_install.py" ;;
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
