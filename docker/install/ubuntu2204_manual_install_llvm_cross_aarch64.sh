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

set -e
set -u
set -o pipefail

architecture_type=$(uname -i)
# Cross-build LLVM for aarch64 when not building natively.
if [ "$architecture_type" != "aarch64" ]; then
  git clone --depth 1 --branch release/15.x https://github.com/llvm/llvm-project.git
  pushd llvm-project

  # First build clang-tblgen and llvm-tblgen
  mkdir build-host
  pushd build-host
  cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="llvm;clang;clang-tools-extra" \
    ../llvm
  ninja clang-tblgen llvm-tblgen
  popd

  # Then cross-compile LLVM for Aarch64
  mkdir build
  pushd build
  CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/llvm-aarch64 \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="llvm;clang" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DLLVM_TABLEGEN=/llvm-project/build-host/bin/llvm-tblgen \
    -DCLANG_TABLEGEN=/llvm-project/build-host/bin/clang-tblgen \
    -DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-linux-gnu \
    -DLLVM_TARGET_ARCH=AArch64 \
    -DCMAKE_CXX_FLAGS='-march=armv8-a -mtune=cortex-a72' \
    ../llvm
  ninja install
  popd
  popd
  rm -rf llvm-project

  # This is a hack. Cross-compiling LLVM with gcc will cause the llvm-config to be an Aarch64 executable.
  # We need it to be x86 to be able to call it when building TVM. We just copy and use the x86 one instead.
  cp /usr/bin/llvm-config /usr/llvm-aarch64/bin/llvm-config
fi
