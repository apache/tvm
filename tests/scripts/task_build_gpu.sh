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

set -euxo pipefail

export SKBUILD_CMAKE_BUILD_TYPE=RelWithDebInfo
export SKBUILD_CMAKE_SOURCE_DIR="$PWD"
export SKBUILD_BUILD_DIR="$PWD/build"
export SKBUILD_BUILD_TARGETS=all
export CMAKE_ARGS="
    -DTVM_VERSION=$(python3 -m setuptools_scm)
    -DTVM_BUILD_PYTHON_MODULE=ON
    -DTVM_TIRX_BUILD_PYTHON_MODULE=ON
    -DTVM_FFI_BUILD_PYTHON_MODULE=ON
    -DTVM_FFI_BUILD_TESTS=OFF
    -DTVM_FFI_ATTACH_DEBUG_SYMBOLS=ON
    -DUSE_CUBLAS=ON
    -DUSE_CUDNN=ON
    -DTVM_TIRX_USE_CUDA=ON
    -DUSE_RPC=ON
    -DUSE_SORT=ON
    -DUSE_BLAS=openblas
    -DCMAKE_CXX_FLAGS=-Werror
    -DUSE_TENSORRT_CODEGEN=ON
    -DHIDE_PRIVATE_SYMBOLS=ON
    -DUSE_CUTLASS=ON
    -DCMAKE_CUDA_ARCHITECTURES=75
    -DUSE_CLML=ON
    -DUSE_CCACHE=OFF
    -DSUMMARIZE=ON
    -DTVM_USE_INSTALLED_TIRX=OFF
    -DBUILD_TESTING=OFF
    -DTVM_TIRX_BUILD_TESTS=OFF
    -DTVM_TIRX_USE_VULKAN=ON
    -DTVM_TIRX_USE_OPENCL=ON
    -DTVM_TIRX_USE_OPENCL_GTEST=/googletest
    -DTVM_TIRX_USE_LLVM='/usr/bin/llvm-config-15 --link-static'
"
uv build --wheel --no-build-isolation --offline --python python3 --out-dir dist ./tirx/3rdparty/tvm-ffi
uv build --wheel --no-build-isolation --offline --python python3 --out-dir dist ./tirx
uv build --wheel --no-build-isolation --offline --python python3
