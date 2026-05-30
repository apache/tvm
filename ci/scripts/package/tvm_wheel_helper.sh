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
TVM_CUDA_RUNTIME_PATH="${TVM_CUDA_RUNTIME_PATH:-}"
TVM_USE_LLVM="${TVM_USE_LLVM:-llvm-config --link-static}"
TVM_USE_CUDA="${TVM_USE_CUDA:-ON}"
TVM_CUDA_ARCHITECTURES="${TVM_CUDA_ARCHITECTURES:-75}"
TVM_BUILD_PARALLEL_LEVEL="${TVM_BUILD_PARALLEL_LEVEL:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}}"
TVM_INCLUDE_CUDA_RUNTIME="${TVM_INCLUDE_CUDA_RUNTIME:-}"

normalize_bool() {
  local name="$1"
  local value="$2"
  local normalized
  normalized="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  case "$normalized" in
    1|true|yes|on) echo 1 ;;
    0|false|no|off) echo 0 ;;
    *)
      echo "error: ${name} must be a boolean value" >&2
      return 1
      ;;
  esac
}

if [[ -n "$TVM_INCLUDE_CUDA_RUNTIME" ]]; then
  tvm_include_cuda_runtime_normalized="$(normalize_bool TVM_INCLUDE_CUDA_RUNTIME "$TVM_INCLUDE_CUDA_RUNTIME")"
  if [[ -n "${TVM_SKIP_CUDA+x}" ]]; then
    tvm_skip_cuda_normalized="$(normalize_bool TVM_SKIP_CUDA "$TVM_SKIP_CUDA")"
    if [[ "$tvm_include_cuda_runtime_normalized" == "$tvm_skip_cuda_normalized" ]]; then
      echo "error: TVM_INCLUDE_CUDA_RUNTIME conflicts with TVM_SKIP_CUDA" >&2
      exit 1
    fi
  fi
  if [[ "$tvm_include_cuda_runtime_normalized" == "1" ]]; then
    TVM_SKIP_CUDA=0
  else
    TVM_SKIP_CUDA=1
  fi
else
  TVM_SKIP_CUDA="${TVM_SKIP_CUDA:-0}"
fi
TVM_SKIP_CUDA="$(normalize_bool TVM_SKIP_CUDA "$TVM_SKIP_CUDA")"
TVM_KEEP_BUILD_DIRS="${TVM_KEEP_BUILD_DIRS:-0}"
TVM_KEEP_BUILD_DIRS="$(normalize_bool TVM_KEEP_BUILD_DIRS "$TVM_KEEP_BUILD_DIRS")"

usage() {
  cat <<'EOF'
Usage: ci/scripts/package/tvm_wheel_helper.sh [cuda|cuda-path|manylinux-cuda|verify-pypi]

The main wheel build, repair, and post-install tests are owned by cibuildwheel
(see pyproject.toml [tool.cibuildwheel] and the publish workflow). This helper
only covers the pieces cibuildwheel cannot: building the CUDA runtime sidecar
and verifying an already-published package.

Environment knobs:
  TVM_USE_LLVM                 LLVM config for the CUDA runtime build, default "llvm-config --link-static"
  TVM_USE_CUDA                 CUDA root or ON for the CUDA build, default ON
  TVM_CUDA_RUNTIME_PATH        Explicit libtvm_runtime_cuda.so path
  TVM_CUDA_ARCHITECTURES       CMake CUDA arch list, default 75
  TVM_INCLUDE_CUDA_RUNTIME=1   Build libtvm_runtime_cuda.so
  TVM_SKIP_CUDA=1              Do not build libtvm_runtime_cuda.so
  TVM_KEEP_BUILD_DIRS=1        Reuse CMake build dirs instead of cleaning them
  TVM_MANYLINUX_IMAGE          manylinux image tag for manylinux-cuda
  TVM_MANYLINUX_IMAGE_TAG      pinned image tag for manylinux-cuda
  TVM_ARCH                     Target architecture for manylinux-cuda
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
  if [[ -n "$TVM_CUDA_RUNTIME_PATH" ]]; then
    if [[ -f "$TVM_CUDA_RUNTIME_PATH" ]]; then
      echo "$TVM_CUDA_RUNTIME_PATH"
    else
      echo "error: TVM_CUDA_RUNTIME_PATH does not exist: ${TVM_CUDA_RUNTIME_PATH}" >&2
      return 1
    fi
    return 0
  fi
  if [[ ! -d "$TVM_CUDA_BUILD_DIR" ]]; then
    return 0
  fi
  find "$TVM_CUDA_BUILD_DIR" -type f -name 'libtvm_runtime_cuda.so' | sort | tail -n 1
}

manylinux_image_name() {
  local base="$1"
  local arch="$2"
  local tag="${3:-}"
  if [[ "$base" == *"/"* || "$base" == *":"* ]]; then
    if [[ "$base" != *@sha256:* && "${base##*/}" != *":"* ]]; then
      echo "error: fully qualified TVM_MANYLINUX_IMAGE must include a tag or digest" >&2
      return 1
    fi
    echo "$base"
  elif [[ -n "$tag" ]]; then
    echo "quay.io/pypa/${base}_${arch}:${tag}"
  else
    echo "error: TVM_MANYLINUX_IMAGE_TAG is required when TVM_MANYLINUX_IMAGE is not fully qualified" >&2
    return 1
  fi
}

validate_manylinux_cuda_image() {
  local image="$1"
  local image_name="${image##*/}"
  if [[ "$image_name" != manylinux_2_28_*:* && "$image_name" != manylinux_2_28_*@sha256:* ]]; then
    echo "error: manylinux-cuda currently supports only pinned manylinux_2_28 images" >&2
    return 1
  fi
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

  local image
  image="$(manylinux_image_name "$TVM_MANYLINUX_IMAGE" "$TVM_ARCH" "${TVM_MANYLINUX_IMAGE_TAG:-}")"
  validate_manylinux_cuda_image "$image"
  local container="tvm_wheel_cuda_${GITHUB_RUN_ID:-local}_${GITHUB_RUN_ATTEMPT:-1}_${TVM_ARCH}"
  local host_cuda_build_dir="$TVM_CUDA_BUILD_DIR"
  local container_cuda_root="/workspace-cuda-build"
  local container_cuda_build_dir="${container_cuda_root}/build"
  mkdir -p "$host_cuda_build_dir"
  local cuda_rpm="/tmp/cuda-repo-rhel8-13-0-local-13.0.2_580.95.05-1.${TVM_ARCH}.rpm"
  trap "rm -f '${cuda_rpm}'; docker exec '${container}' bash -lc 'chown -R $(id -u):$(id -g) ${container_cuda_root} || true' >/dev/null 2>&1 || true; docker rm -f '${container}' >/dev/null 2>&1 || true" EXIT
  docker pull "$image"
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker run --name "$container" -d \
    --workdir /workspace \
    --volume "${REPO_ROOT}:/workspace" \
    --volume "${host_cuda_build_dir}:${container_cuda_root}" \
    "$image" tail -f /dev/null

  local cuda_rpm_name
  cuda_rpm_name="$(basename "$cuda_rpm")"
  curl -fsSLo "$cuda_rpm" "https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/${cuda_rpm_name}"
  docker cp "$cuda_rpm" "${container}:/${cuda_rpm_name}"
  rm "$cuda_rpm"
  docker exec "$container" bash -lc "
    rpm -i /${cuda_rpm_name} && \
    dnf clean all && \
    dnf -y --disablerepo=epel install cuda-toolkit-13-0 && \
    rm /${cuda_rpm_name} && \
    dnf clean all"

  docker exec \
    -e TVM_PYTHON=/opt/python/cp310-cp310/bin/python \
    -e TVM_USE_CUDA=/usr/local/cuda \
    -e TVM_CUDA_ARCHITECTURES="$TVM_CUDA_ARCHITECTURES" \
    -e TVM_CUDA_BUILD_DIR="$container_cuda_build_dir" \
    -e TVM_INCLUDE_CUDA_RUNTIME=1 \
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
    "chown -R $(id -u):$(id -g) ${container_cuda_root} || true"
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
  "$venv_python" -m pip install pytest numpy
  "$venv_python" -m pytest "$REPO_ROOT/tests/python/wheel"
}

main() {
  local step="${1:-help}"
  case "$step" in
    cuda) build_cuda_runtime ;;
    cuda-path) cuda_runtime_path ;;
    manylinux-cuda) run_manylinux_cuda_container ;;
    verify-pypi) verify_pypi_wheel ;;
    -h|--help|help) usage ;;
    *)
      usage >&2
      return 1
      ;;
  esac
}

main "$@"
