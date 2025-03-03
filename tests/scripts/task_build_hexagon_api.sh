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

output_directory_parent=$(realpath ${PWD}/build)
if [ $# -ge 1 ] && [[ "$1" == "--output" ]]; then
    shift 1
    output_directory_parent=$(realpath $1)
    shift 1
fi
output_directory="${output_directory_parent}/hexagon_api_output"
rm -rf ${output_directory}

use_cache=false
if [ $# -ge 1 ] && [[ "$1" == "--use-cache" ]]; then
    use_cache=true
    shift 1
fi

cd apps/hexagon_api
if [ "$use_cache" = false ]; then
    rm -rf build
fi
mkdir -p build
cd build

cmake -DUSE_HEXAGON_ARCH=v68 \
    -DUSE_HEXAGON_SDK="${HEXAGON_SDK_ROOT}" \
    -DUSE_HEXAGON_TOOLCHAIN="${HEXAGON_TOOLCHAIN}" \
    -DUSE_OUTPUT_BINARY_DIR="${output_directory}" \
    -DUSE_HEXAGON_GTEST="${HEXAGON_SDK_ROOT}/utils/googletest/gtest" ..

make -j$(nproc)
