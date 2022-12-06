#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
ubuntu_install_spike_sim.sh

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

function show_usage() {
    cat <<EOF
Usage: docker/install/ubuntu_download_xuantie_gcc_linux.sh <INSTALLATION_PATH>
INSTALLATION_PATH is the installation path for the toolchain.
EOF
}

if [ "$#" -lt 1 -o "$1" == "--help" -o "$1" == "-h" ]; then
    show_usage
    exit 1
fi

INSTALLATION_PATH=$1
shift

# Create installation path directory
mkdir -p "${INSTALLATION_PATH}"

# Download and extract RISC-V gcc
RISCV_GCC_VERSION="2.6.0"
RISCV_GCC_ID="1659325511536"
RISCV_GCC_KERNEL_VERSION="5.10.4"
RISCV_GCC_DATE="20220715"
RISCV_GCC_ARCH="x86_64"
RISCV_GCC_BASE="Xuantie-900-gcc-linux-${RISCV_GCC_KERNEL_VERSION}-glibc-${RISCV_GCC_ARCH}-V${RISCV_GCC_VERSION}-${RISCV_GCC_DATE}"
RISCV_GCC_EXT="tar.gz"
RISCV_GCC_URL="https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//${RISCV_GCC_ID}/${RISCV_GCC_BASE}.${RISCV_GCC_EXT}"
DOWNLOAD_PATH="/tmp/${RISCV_GCC_BASE}.${RISCV_GCC_EXT}"

wget ${RISCV_GCC_URL} -O "${DOWNLOAD_PATH}"
tar -xf "${DOWNLOAD_PATH}" -C "${INSTALLATION_PATH}" --strip-components=1
rm $DOWNLOAD_PATH
echo "SUCCESS"
