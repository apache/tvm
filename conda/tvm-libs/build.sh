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

if [ "$cuda" == "True" ]; then
    CUDA_OPT="-DUSE_CUDA=ON -DUSE_CUBLAS=ON -DUSE_CUDNN=ON"
else
    CUDA_OPT=""
fi

if [ "$target_platform" == "osx-64" ]; then
    # macOS 64 bits
    METAL_OPT=""  # Conda can only target 10.9 for now
else
    METAL_OPT=""
fi

rm -rf build || true
mkdir -p build
cd build
cmake $METAL_OPT $CUDA_OPT -DUSE_LLVM=$PREFIX/bin/llvm-config -DINSTALL_DEV=ON -DCMAKE_INSTALL_PREFIX="$PREFIX" ..
make -j${CPU_COUNT} VERBOSE=1
make install
cd ..
