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
# CPU-only wheels. This is the only piece cibuildwheel cannot do itself, since
# its manylinux container ships no CUDA toolkit.
#
# Usage: before_all_linux.sh <include_cuda_runtime: 0|1> <cuda_architectures>
set -euxo pipefail

include_cuda_runtime="${1:-0}"
cuda_architectures="${2:-75}"

if [[ "${include_cuda_runtime}" != "1" ]]; then
  echo "before_all_linux: CUDA runtime not requested; nothing to do."
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
build_dir="${repo_root}/build-wheel-cuda"
python_bin="/opt/python/cp310-cp310/bin/python"
parallel="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

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

# Build the CUDA runtime sidecar with CUDA on and LLVM off, so it does not need
# the LLVM prefix; the main CPU wheel links LLVM statically. before-all runs
# before CIBW_BEFORE_BUILD, so install the build tools here.
export PATH="/opt/python/cp310-cp310/bin:/usr/local/cuda/bin:${PATH}"
"${python_bin}" -m pip install -U pip cmake ninja
nvcc --version

rm -rf "${build_dir}"
cmake -S "${repo_root}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DTVM_BUILD_PYTHON_MODULE=ON \
  -DUSE_CUDA=/usr/local/cuda \
  -DUSE_LLVM=OFF \
  -DUSE_CUBLAS=OFF -DUSE_CUDNN=OFF -DUSE_CUTLASS=OFF -DUSE_NCCL=OFF -DUSE_NVTX=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build "${build_dir}" --target tvm_runtime tvm_runtime_cuda --parallel "${parallel}"

cuda_lib="${build_dir}/lib/libtvm_runtime_cuda.so"
test -f "${cuda_lib}"
patchelf --set-rpath '$ORIGIN' "${cuda_lib}"
echo "CUDA runtime: ${cuda_lib}"
