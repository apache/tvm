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

install_path="/opt/csi-nn2"

# Clone CSI-NN2 Compute Library source code
git clone --depth 1 --branch 1.12.2 https://github.com/T-head-Semi/csi-nn2.git ${install_path}

# download cross-compiler when not building natively.
# riscv gcc toolchain will be downloaded to "/path/csi-nn2/tools/gcc-toolchain".
cd ${install_path}
./script/download_toolchain.sh

# download custom QEMU to "/path/csi-nn2/tools/qemu".
./script/download_qemu.sh

# build csinn2 lib for x86 and c906
# lib will be installed in /path/csi-nn2/install
# for x86
make -j4; cd x86_build; make install; cd -
# for c906
mkdir -p riscv_build; cd riscv_build
cmake ../ -DBUILD_RISCV=ON; make -j4; make install; cd -

