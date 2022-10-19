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
CLANG_LLVM_VERSION=14.0.0
CLANG_LLVM_FILENAME=clang_llvm.tar.xz
wget -q https://github.com/llvm/llvm-project/releases/download/llvmorg-${CLANG_LLVM_VERSION}/clang+llvm-${CLANG_LLVM_VERSION}-x86_64-linux-gnu-ubuntu-18.04.tar.xz -O ${CLANG_LLVM_FILENAME}
mkdir ${CLANG_LLVM_HOME}
tar -xvf ${CLANG_LLVM_FILENAME} -C ${CLANG_LLVM_HOME} --strip-components=1
rm ${CLANG_LLVM_FILENAME}
