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

fvp_dir="/opt/arm/FVP_Corstone_SSE-300"
cmake_dir="/opt/arm/cmake"
ethosu_dir="/opt/arm/ethosu"
ethosu_driver_ver="21.11"

mkdir -p /opt/arm

tmpdir=$(mktemp -d)

cleanup()
{
  rm -rf "$tmpdir"
}

trap cleanup 0

# Ubuntu 18.04 dependencies
apt-get update
apt-install-and-clear -y \
    bsdmainutils \
    build-essential \
    cpp \
    git \
    linux-headers-generic \
    make \
    python-dev \
    python3 \
    ssh \
    wget \
    xxd

# Download the FVP
mkdir -p "$fvp_dir"
cd "$tmpdir"
curl -sL https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_11.15_24.tgz | tar -xz
./FVP_Corstone_SSE-300.sh --i-agree-to-the-contained-eula --no-interactive -d "$fvp_dir"
rm -rf FVP_Corstone_SSE-300.sh license_terms

# Setup cmake 3.19.5
mkdir -p "${cmake_dir}"
cd "$tmpdir"
curl -sL -o cmake-3.19.5-Linux-x86_64.sh https://github.com/Kitware/CMake/releases/download/v3.19.5/cmake-3.19.5-Linux-x86_64.sh
chmod +x cmake-3.19.5-Linux-x86_64.sh
./cmake-3.19.5-Linux-x86_64.sh --prefix="${cmake_dir}" --skip-license
rm cmake-3.19.5-Linux-x86_64.sh
export PATH="${cmake_dir}/bin:${PATH}"

# Install the GCC toolchain
mkdir -p /opt/arm/gcc-arm-none-eabi/
gcc_arm_url='https://developer.arm.com/-/media/Files/downloads/gnu-rm/10-2020q4/gcc-arm-none-eabi-10-2020-q4-major-x86_64-linux.tar.bz2?revision=ca0cbf9c-9de2-491c-ac48-898b5bbc0443&la=en&hash=68760A8AE66026BCF99F05AC017A6A50C6FD832A'
curl --retry 64 -sSL ${gcc_arm_url} | tar -C /opt/arm/gcc-arm-none-eabi --strip-components=1 -jx
export PATH="/opt/arm/gcc-arm-none-eabi/bin:${PATH}"

# Clone Arm(R) Ethos(TM)-U NPU driver stack
mkdir -p "${ethosu_dir}"
cd "${ethosu_dir}"
git clone --branch ${ethosu_driver_ver} "https://review.mlplatform.org/ml/ethos-u/ethos-u-core-driver" core_driver
git clone --branch ${ethosu_driver_ver} "https://review.mlplatform.org/ml/ethos-u/ethos-u-core-platform" core_platform

# Build Driver
mkdir ${ethosu_dir}/core_driver/build && cd ${ethosu_dir}/core_driver/build
cmake -DCMAKE_TOOLCHAIN_FILE=${ethosu_dir}/core_platform/cmake/toolchain/arm-none-eabi-gcc.cmake -DETHOSU_LOG_SEVERITY=debug -DTARGET_CPU=cortex-m55 ..
make

# Build NN Library
mkdir ${CMSIS_PATH}/CMSIS/NN/build/ && cd ${CMSIS_PATH}/CMSIS/NN/build/
cmake .. -DCMAKE_TOOLCHAIN_FILE=${ethosu_dir}/core_platform/cmake/toolchain/arm-none-eabi-gcc.cmake -DTARGET_CPU=cortex-m55 -DBUILD_CMSIS_NN_FUNCTIONS=YES
make
