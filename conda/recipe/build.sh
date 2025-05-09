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

GPU_OPT=""
TOOLCHAIN_OPT=""
MACOS_OPT=""

if [ "$target_platform" == "osx-64" ]; then
    # macOS 64 bits
    GPU_OPT="-DUSE_METAL=ON"
    MACOS_OPT="-DCMAKE_OSX_DEPLOYMENT_TARGET=10.13"
elif [ "$target_platform" == "linux-64" ]; then
    TOOLCHAIN_OPT="-DCMAKE_TOOLCHAIN_FILE=${RECIPE_DIR}/cross-linux.cmake"
fi

# When cuda is not set, we default to False
cuda=${cuda:-False}

if [ "$cuda" == "True" ]; then
    GPU_OPT="-DUSE_CUDA=ON -DUSE_CUBLAS=ON -DUSE_CUDNN=ON"
    TOOLCHAIN_OPT=""
fi

# remove touched cmake config
rm -f config.cmake
rm -rf build || true
mkdir -p build
cd build

cmake -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_RPC=ON \
      -DUSE_CPP_RPC=OFF \
      -DUSE_SORT=ON \
      -DUSE_RANDOM=ON \
      -DUSE_PROFILER=ON \
      -DUSE_LLVM=ON \
      -DINSTALL_DEV=ON \
      -DUSE_LIBBACKTRACE=AUTO \
      ${GPU_OPT} ${TOOLCHAIN_OPT} ${MACOS_OPT} \
      ${SRC_DIR}

make -j${CPU_COUNT} VERBOSE=1
cd ..
