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

set -euxo pipefail

BUILD_DIR=$1
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cp ../cmake/config.cmake .

echo set\(USE_SORT ON\) >> config.cmake
echo set\(USE_PROFILER ON\) >> config.cmake
echo set\(USE_DNNL ON\) >> config.cmake
echo set\(USE_ARM_COMPUTE_LIB ON\) >> config.cmake
echo set\(USE_LLVM \"/usr/bin/llvm-config-17 --link-static\"\) >> config.cmake
echo set\(USE_ANTLR ON\) >> config.cmake
echo set\(CMAKE_CXX_FLAGS \"-Werror -Wno-error=range-loop-construct\"\) >> config.cmake
echo set\(HIDE_PRIVATE_SYMBOLS ON\) >> config.cmake

# This conditional is just to support the transition to cope
# with the change in the way TFLite is built. It can be
# removed once we migrate to TensorFlow and TFLite > 2.9.1
if [ -d "/opt/tflite" ]; then
  echo set\(USE_TFLITE \"/opt/tflite\"\) >> config.cmake
else
  echo set\(USE_TFLITE ON\) >> config.cmake
fi

echo set\(USE_TENSORFLOW_PATH \"/tensorflow\"\) >> config.cmake
echo set\(USE_FLATBUFFERS_PATH \"/flatbuffers\"\) >> config.cmake
echo set\(USE_VITIS_AI ON\) >> config.cmake
echo set\(USE_LIBBACKTRACE COMPILE\) >> config.cmake
echo set\(BACKTRACE_ON_SEGFAULT ON\) >> config.cmake
echo set\(USE_CCACHE OFF\) >> config.cmake
echo set\(USE_UMA ON\) >> config.cmake
echo set\(SUMMARIZE ON\) >> config.cmake
echo set\(USE_MSC ON\) >> config.cmake
