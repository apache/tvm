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

cd apps/hexagon_api
rm -rf build
mkdir -p build
cd build

output_binary_directory=$(realpath ${PWD}/../../../build/hexagon_api_output)
rm -rf ${output_binary_directory}

cmake -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DUSE_ANDROID_TOOLCHAIN="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
    -DUSE_HEXAGON_ARCH=v68 \
    -DUSE_HEXAGON_SDK="${HEXAGON_SDK_PATH}" \
    -DUSE_HEXAGON_TOOLCHAIN="${HEXAGON_TOOLCHAIN}" \
    -DUSE_OUTPUT_BINARY_DIR="${output_binary_directory}" ..

make -j$(nproc)
