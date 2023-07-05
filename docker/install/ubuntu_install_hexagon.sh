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

set -o errexit -o nounset
set -o pipefail

# Install LLVM/clang
CLANG_LLVM_HOME=/opt/clang-llvm
LLVM_SHA=a9871772a8b13c1240a95a84a3327f84bb67dddc

mkdir llvm-hexagon
pushd llvm-hexagon
git init
git remote add origin https://github.com/llvm/llvm-project.git
git fetch origin ${LLVM_SHA}
git reset --hard FETCH_HEAD
mkdir build
pushd build
cmake \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${CLANG_LLVM_HOME} \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD:STRING="Hexagon;X86" \
  -DLLVM_ENABLE_PROJECTS:STRING="llvm" \
  -DLLVM_DEFAULT_TARGET_TRIPLE=x86_64-unknown-linux-gnu \
  ../llvm
ninja install

popd
popd
rm -rf llvm-hexagon
