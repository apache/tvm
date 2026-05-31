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
#
# cibuildwheel CIBW_BEFORE_ALL_LINUX hook. It runs once per architecture inside
# the manylinux build container, before any wheel is built. When a CUDA runtime
# is requested it installs the pinned CUDA toolkit and builds
# libtvm_runtime_cuda.so into build-wheel-cuda/, so the subsequent (CPU-only)
# wheel build can bundle it via -DTVM_PACKAGE_EXTRA_LIBS. It is a no-op for
# CPU-only wheels. This replaces the previous separate docker-in-docker
# build-cuda action; everything now happens in the single cibuildwheel container.
#
# Usage: before_all_linux.sh <include_cuda_runtime: 0|1> <cuda_architectures>
set -euxo pipefail

include_cuda_runtime="${1:-0}"
cuda_architectures="${2:-75}"

if [[ "${include_cuda_runtime}" != "1" ]]; then
  echo "before_all_linux: CUDA runtime not requested; nothing to do."
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install the pinned CUDA toolkit into the manylinux_2_28 container. The RHEL8
# local-repo RPM is compatible with manylinux_2_28 for both x86_64 and aarch64.
arch="$(uname -m)"
cuda_rpm="cuda-repo-rhel8-13-0-local-13.0.2_580.95.05-1.${arch}.rpm"
curl -fsSLo "/tmp/${cuda_rpm}" \
  "https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/${cuda_rpm}"
rpm -i "/tmp/${cuda_rpm}"
dnf clean all
dnf -y --disablerepo=epel install cuda-toolkit-13-0
rm -f "/tmp/${cuda_rpm}"
dnf clean all

# Build libtvm_runtime_cuda.so with a manylinux CPython and a pip-installed cmake
# (before-all runs before CIBW_BEFORE_BUILD, so the build tools are not yet present).
python_bin="/opt/python/cp310-cp310/bin/python"
export PATH="/opt/python/cp310-cp310/bin:/usr/local/cuda/bin:${PATH}"
"${python_bin}" -m pip install -U pip cmake ninja
nvcc --version

TVM_PYTHON="${python_bin}" \
TVM_USE_CUDA="/usr/local/cuda" \
TVM_USE_LLVM="OFF" \
TVM_CUDA_BUILD_DIR="${TVM_CUDA_BUILD_DIR:-$(pwd)/build-wheel-cuda}" \
TVM_CUDA_ARCHITECTURES="${cuda_architectures}" \
TVM_INCLUDE_CUDA_RUNTIME="1" \
  "${script_dir}/tvm_wheel_helper.sh" cuda
