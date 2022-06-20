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

echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main\
     >> /etc/apt/sources.list.d/llvm.list


echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-13 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-13 main\
     >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic main\
     >> /etc/apt/sources.list.d/llvm.list

wget -q -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -

apt-get update && apt-install-and-clear -y \
     llvm-9 llvm-10 llvm-11 llvm-12 llvm-13 \
     clang-9 libclang-9-dev \
     clang-10 libclang-10-dev \
     clang-11 libclang-11-dev \
     clang-12 libclang-12-dev \
     clang-13 libclang-13-dev
