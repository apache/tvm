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

cd /usr
git clone --depth=1 https://github.com/dmlc/tvm --recursive
cd /usr/tvm
echo set\(USE_LLVM llvm-config-6.0\) >> config.cmake
echo set\(USE_RPC ON\) >> config.cmake
echo set\(USE_SORT ON\) >> config.cmake
echo set\(USE_GRAPH_RUNTIME ON\) >> config.cmake
echo set\(USE_BLAS openblas\) >> config.cmake
echo set\(USE_SGX /opt/sgxsdk\) >> config.cmake
echo set\(RUST_SGX_SDK /opt/rust-sgx-sdk\) >> config.cmake
mkdir -p build
cd build
cmake ..
make -j10
