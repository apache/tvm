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

set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}
: ${WORKSPACE_CWD:=$(pwd)}
: ${GPU:="cpu"}
export CCACHE_COMPILERCHECK=content
export CCACHE_NOHASHDIR=1
export CCACHE_DIR=/ccache-tvm

source /multibuild/manylinux_utils.sh
source /opt/rh/gcc-toolset-11/enable # GCC-11 is the hightest GCC version compatible with NVCC < 12

rm -rf $WORKSPACE_CWD/build/
mkdir -p $WORKSPACE_CWD/build/ && cd $WORKSPACE_CWD/build/
cp $WORKSPACE_CWD/cmake/config.cmake $WORKSPACE_CWD/build/ -f

echo set\(USE_OPENCL ON\) >> config.cmake
echo set\(USE_LLVM ON\) >> config.cmake
if [[ ${GPU} == cuda* ]]; then
    echo set\(CMAKE_CUDA_COMPILER_LAUNCHER ccache\) >>config.cmake
    echo set\(CMAKE_CUDA_ARCHITECTURES "80;90"\) >>config.cmake
    echo set\(CMAKE_CUDA_FLAGS \"\$\{CMAKE_CUDA_FLAGS\} -t $NUM_THREADS\"\) >>config.cmake
    echo set\(USE_CUDA ON\) >> config.cmake
fi
if [[ -f ${ADRENOACCL_SDK}/include/adrenoaccl.h ]]; then
    echo set\(USE_ADRENO_ACCL \"${ADRENOACCL_SDK}\"\) >> config.cmake
fi
if [[ -f ${ADRENO_OPENCL}/CL/cl.h ]]; then
    echo set\(USE_CLML \"${ADRENO_OPENCL}\"\) >> config.cmake
fi

cmake .. && make -j`nproc`
