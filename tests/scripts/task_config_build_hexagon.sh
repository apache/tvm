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

BUILD_DIR=$1
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cp ../cmake/config.cmake .

echo set\(USE_SORT ON\) >> config.cmake
echo set\(USE_RPC ON\) >> config.cmake
echo set\(USE_MICRO ON\) >> config.cmake
echo set\(USE_MICRO_STANDALONE_RUNTIME ON\) >> config.cmake
echo set\(USE_LLVM "${CLANG_LLVM_HOME}/bin/llvm-config"\) >> config.cmake
echo set\(CMAKE_CXX_COMPILER "${CLANG_LLVM_HOME}/bin/clang++"\) >> config.cmake
echo set\(USE_HEXAGON "ON"\) >> config.cmake
echo set\(USE_HEXAGON_SDK "${HEXAGON_SDK_PATH}"\) >> config.cmake
echo set\(USE_CCACHE OFF\) >> config.cmake
echo set\(SUMMARIZE ON\) >> config.cmake
