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

RISCV_TOOLCHAIN_VERSION="2023.07.07"
SPIKE_VERSION="v1.1.0"
TMPDIR=$(mktemp -d)
INSTALLATION_PATH=$1
export PATH=$INSTALLATION_PATH/bin:$PATH
shift

# Install the packages required to build the toolchain
sudo apt-install-and-clear -y --no-install-recommends device-tree-compiler autoconf automake autotools-dev curl libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git libglib2.0-dev

# Create installation path directory
mkdir -p "${INSTALLATION_PATH}"

# Install RISC-V GNU toolchain
cd $TMPDIR
git clone https://github.com/riscv/riscv-gnu-toolchain -b $RISCV_TOOLCHAIN_VERSION
pushd riscv-gnu-toolchain
./configure --prefix=$INSTALLATION_PATH --enable-multilib
make linux
popd

# Install spike
git clone https://github.com/riscv/riscv-isa-sim.git -b $SPIKE_VERSION
pushd riscv-isa-sim
mkdir build
cd build
../configure --prefix=$INSTALLATION_PATH
make -j`nproc`
make install
popd

# Install pk
git clone https://github.com/riscv/riscv-pk.git
pushd riscv-pk

# rv32gc
mkdir build
pushd build
../configure --prefix=`pwd`/install --host=riscv64-unknown-linux-gnu --with-arch=rv32gc
make -j`nproc`
make install
cp ./pk $INSTALLATION_PATH/riscv64-unknown-linux-gnu/bin/pk
popd

# rv64gc
mkdir build64
pushd build64
../configure --prefix=`pwd`/install --host=riscv64-unknown-linux-gnu --with-arch=rv64gc
make -j`nproc`
make install
cp ./pk $INSTALLATION_PATH/riscv64-unknown-linux-gnu/bin/pk64

# cleanup
rm -rf $TMPDIR

echo "SUCCESS"
