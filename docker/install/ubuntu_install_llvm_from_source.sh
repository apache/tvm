#!/bin/bash

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
# This script builds LLVM and clang from the llvm-project tarball
# using CMake. It is tested with LLVM from version 15.

set -e

LLVM_VERSION_MAJOR=$1

detect_llvm_version() {
  curl -sL "https://api.github.com/repos/llvm/llvm-project/releases?per_page=100" | \
    grep tag_name | \
    grep -o "llvmorg-${LLVM_VERSION_MAJOR}[^\"]*" | \
    grep -v rc | \
    sed -e "s/^llvmorg-//g" | \
    head -n 1
}

LLVM_VERSION=$(detect_llvm_version)
echo ${LLVM_VERSION}

curl -sL \
  https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz \
  -o llvm-project-${LLVM_VERSION}.src.tar.xz
unxz llvm-project-${LLVM_VERSION}.src.tar.xz
tar xf llvm-project-${LLVM_VERSION}.src.tar
pushd llvm-project-${LLVM_VERSION}.src

pushd llvm
mkdir build
pushd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MODULE_PATH="/llvm-project-${LLVM_VERSION}.src/cmake/Modules" \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DLLVM_TARGETS_TO_BUILD="AArch64;ARM;X86" \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_UTILS=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_USE_INTEL_JITEVENTS=ON \
    -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON \
    -DPYTHON_EXECUTABLE="$(cpython_path 3.7)/bin/python" \
    -GNinja \
    ..
ninja install
popd
popd
rm -rf llvm-${LLVM_VERSION}.src.tar.xz llvm-${LLVM_VERSION}.src.tar llvm-${LLVM_VERSION}.src

# clang is only used to precompile Gandiva bitcode
if [ ${LLVM_VERSION_MAJOR} -lt 9 ]; then
  clang_package_name=cfe
else
  clang_package_name=clang
fi

pushd ${clang_package_name}
mkdir build
pushd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_MODULE_PATH="/llvm-project-${LLVM_VERSION}.src/cmake/Modules" \
    -DCLANG_INCLUDE_TESTS=OFF \
    -DCLANG_INCLUDE_DOCS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -Wno-dev \
    -GNinja \
    ..

ninja -w dupbuild=warn install # both clang and llvm builds generate llvm-config file
popd
popd

# out of llvm-project-${LLVM_VERSION}.src
popd
rm -rf ${clang_package_name}-${LLVM_VERSION}.src.tar.xz ${clang_package_name}-${LLVM_VERSION}.src.tar ${clang_package_name}-${LLVM_VERSION}.src
