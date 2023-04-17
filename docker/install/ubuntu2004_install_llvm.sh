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

set -euxo pipefail
apt-get update
apt-install-and-clear -y gnupg

echo deb http://apt.llvm.org/focal/ llvm-toolchain-focal-13 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-13 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/focal/ llvm-toolchain-focal-15 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-15 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/focal/ llvm-toolchain-focal main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal main\
     >> /etc/apt/sources.list.d/llvm.list

wget -q -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
apt-get update && apt-install-and-clear -y llvm-15 llvm-14 llvm-13 \
    clang-15 libclang-15-dev \
    clang-14 libclang-14-dev \
    clang-13 libclang-13-dev
