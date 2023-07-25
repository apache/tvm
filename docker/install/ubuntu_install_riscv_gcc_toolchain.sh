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

INSTALLATION_PATH=$1
shift

# Create installation path directory
mkdir -p "${INSTALLATION_PATH}"

# Download and extract RISC-V gcc
TMPDIR=$(mktemp -d)
RISCV_GCC_EXT="tar.gz"
RLS_TAG="2023.07.07"
RISCV_GCC_URL="https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/${RLS_TAG}/riscv64-glibc-ubuntu-22.04-gcc-nightly-${RLS_TAG}-nightly.${RISCV_GCC_EXT}"

DOWNLOAD_PATH="$TMPDIR/${RLS_TAG}.${RISCV_GCC_EXT}"

wget ${RISCV_GCC_URL} -O "${DOWNLOAD_PATH}"
tar -xf "${DOWNLOAD_PATH}" -C "${INSTALLATION_PATH}" --strip-components=1

export PATH=$INSTALLATION_PATH/bin:$PATH

sudo apt-install-and-clear -y --no-install-recommends device-tree-compiler

# Install spike
mkdir $TMPDIR/spike
cd $TMPDIR/spike
git clone https://github.com/riscv/riscv-isa-sim.git
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
mkdir build
pushd build
../configure --prefix=`pwd`/install --host=riscv64-unknown-linux-gnu
make -j`nproc`
make install
cp ./pk $INSTALLATION_PATH/bin/pk

# cleanup
rm -rf $TMPDIR

echo "SUCCESS"
