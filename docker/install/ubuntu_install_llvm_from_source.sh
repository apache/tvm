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

LLVM_VERSION=$1
LLVM_FILE_SHA=$2

echo ${LLVM_VERSION}

tmpdir=$(mktemp -d)

cleanup()
{
    rm -rf "$tmpdir"
}

trap cleanup 0

pushd "$tmpdir"

curl -sL \
  https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz \
  -o llvm-project-${LLVM_VERSION}.src.tar.xz
echo "$LLVM_FILE_SHA llvm-project-${LLVM_VERSION}.src.tar.xz" | sha256sum --check
tar xf llvm-project-${LLVM_VERSION}.src.tar.xz
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
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_USE_INTEL_JITEVENTS=ON \
    -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON \
    -DPYTHON_EXECUTABLE="$(which python3.9)" \
    -GNinja \
    ..
ninja install
popd
popd

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
popd
