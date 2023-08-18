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

function show_usage() {
    cat <<EOF
Usage: docker/install/ubuntu_install_riscv_gcc_toolchain.sh <INSTALLATION_PATH>
INSTALLATION_PATH is the installation path for the toolchain.
EOF
}

if [ "$#" -lt 1 -o "$1" == "--help" -o "$1" == "-h" ]; then
    show_usage
    exit 1
fi

RISCV_TOOLCHAIN_VERSION="2023.01.04"
SPIKE_VERSION="v1.1.0"
TMPDIR=$(mktemp -d)
INSTALLATION_PATH=$1
INSTALLATION_PATH_32=${INSTALLATION_PATH}/32
INSTALLATION_PATH_64=${INSTALLATION_PATH}/64
shift

# Install the packages required to build the toolchain
sudo apt-install-and-clear -y --no-install-recommends device-tree-compiler

# Create installation path directories
mkdir -p "${INSTALLATION_PATH_32}"
mkdir -p "${INSTALLATION_PATH_64}"
cd $TMPDIR

# Install RISC-V GNU toolchain
# 32
DOWNLOAD_PATH="$TMPDIR/${RISCV_TOOLCHAIN_VERSION}_32.tar.gz"
wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2023.01.04/riscv32-glibc-ubuntu-22.04-nightly-2023.01.04-nightly.tar.gz -O "${DOWNLOAD_PATH}"
tar -xf "${DOWNLOAD_PATH}" -C "${INSTALLATION_PATH_32}" --strip-components=1
# 64
DOWNLOAD_PATH="$TMPDIR/${RISCV_TOOLCHAIN_VERSION}_64.tar.gz"
wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2023.01.04/riscv64-glibc-ubuntu-22.04-nightly-2023.01.04-nightly.tar.gz -O "${DOWNLOAD_PATH}"
tar -xf "${DOWNLOAD_PATH}" -C "${INSTALLATION_PATH_64}" --strip-components=1

# Install spike
git clone https://github.com/riscv/riscv-isa-sim.git -b $SPIKE_VERSION
pushd riscv-isa-sim
# 32
mkdir build
pushd build
../configure --prefix=$INSTALLATION_PATH_32
make -j`nproc`
make install
popd
# 64
mkdir build64
cd build64
../configure --prefix=$INSTALLATION_PATH_64
make -j`nproc`
make install
popd

# Install pk
git clone https://github.com/riscv/riscv-pk.git
pushd riscv-pk
SRC_PATH=$PATH
# rv32gc
export PATH=$INSTALLATION_PATH_32/bin:$SRC_PATH
mkdir build
pushd build
../configure --prefix=`pwd`/install --host=riscv32-unknown-linux-gnu --with-arch=rv32gc --with-abi=ilp32d
make -j`nproc`
make install
cp ./pk $INSTALLATION_PATH_32/riscv32-unknown-linux-gnu/bin/pk
popd
# rv64gc
export PATH=$INSTALLATION_PATH_64/bin:$SRC_PATH
mkdir build64
pushd build64
../configure --prefix=`pwd`/install --host=riscv64-unknown-linux-gnu --with-arch=rv64gc --with-abi=lp64d
make -j`nproc`
make install
cp ./pk $INSTALLATION_PATH_64/riscv64-unknown-linux-gnu/bin/pk

# cleanup
rm -rf $TMPDIR

echo "SUCCESS"
