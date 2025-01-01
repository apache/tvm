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
set -x

output_directory=$(realpath ${PWD}/build-adreno-target)
rm -rf ${output_directory}

mkdir -p ${output_directory}
cd ${output_directory}

cp ../cmake/config.cmake .

if [ -f "${ADRENO_OPENCL}/CL/cl_qcom_ml_ops.h" ] ; then
echo set\(USE_CLML "${ADRENO_OPENCL}"\) >> config.cmake
echo set\(USE_CLML_GRAPH_EXECUTOR "${ADRENO_OPENCL}"\) >> config.cmake
fi
if [ -f "${ADRENO_OPENCL}/CL/cl.h" ] ; then
echo set\(USE_OPENCL "${ADRENO_OPENCL}"\) >> config.cmake
else
echo set\(USE_OPENCL ON\) >> config.cmake
fi
echo set\(USE_RPC ON\) >> config.cmake
echo set\(USE_CPP_RPC ON\) >> config.cmake
echo set\(USE_CPP_RTVM ON\) >> config.cmake
echo set\(USE_GRAPH_EXECUTOR ON\) >> config.cmake
echo set\(USE_LIBBACKTRACE AUTO\) >> config.cmake
echo set\(USE_KALLOC_ALIGNMENT 32\) >> config.cmake

echo set\(ANDROID_ABI arm64-v8a\) >> config.cmake
echo set\(ANDROID_PLATFORM android-28\) >> config.cmake
echo set\(MACHINE_NAME aarch64-linux-gnu\) >> config.cmake

echo set\(USE_OPENCL_GTEST ON\) >> config.cmake

echo set\(USE_OPENCL_EXTN_QCOM ON\) >> config.cmake

cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-28 \
      -DCMAKE_SYSTEM_VERSION=1 \
      -DCMAKE_FIND_ROOT_PATH="${ADRENO_OPENCL}" \
      -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
      -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
      -DCMAKE_CXX_COMPILER="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++" \
      -DCMAKE_C_COMPILER="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang" \
      -DMACHINE_NAME="aarch64-linux-gnu" ..

make -j$(nproc) tvm_rpc rtvm opencl-cpptest
