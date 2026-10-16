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

# This file is a compiler test to ensure that runtimes can compile
# correctly, even if they aren't actively tested in the CI.
set -euxo pipefail

# Preserve the separate runtime-only GPU compile check.
uv pip install --python python3 --no-deps --target build/python ./3rdparty/tvm-ffi
export PYTHONPATH="$PWD/build/python"
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DTVM_TIRX_USE_OPENCL=ON -DTVM_TIRX_USE_ROCM=ON -DCMAKE_CXX_FLAGS=-Werror
cmake --build build --target tvm_runtime tvm_runtime_opencl tvm_runtime_rocm
