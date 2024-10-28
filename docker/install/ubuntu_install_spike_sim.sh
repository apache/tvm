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
set -x

function show_usage() {
    cat <<EOF
Usage: docker/install/ubuntu_install_spike_sim.sh <RISCV_PATH>
RISCV_PATH is the installation path of the risc-v gcc.
EOF
}

if [ "$#" -lt 1 -o "$1" == "--help" -o "$1" == "-h" ]; then
    show_usage
    exit -1
fi

export RISCV=$1
export PATH=$RISCV/bin:$PATH
shift

# Install dependency
apt-install-and-clear -y --no-install-recommends device-tree-compiler

# Install spike
mkdir /tmp/spike
pushd /tmp/spike
    # TODO: freeze version?
    git clone https://github.com/riscv/riscv-isa-sim.git
    pushd riscv-isa-sim
        mkdir build
        cd build
        ../configure --prefix=$RISCV --with-isa=RV32IMAC
        make -j`nproc`
        make install
    popd

    # Install pk
    git clone https://github.com/riscv/riscv-pk.git
    pushd riscv-pk
        # With commit 47a2e87, we get the below compilation so we'll use the specific commit
        #   ../pk/pk.c: Assembler messages:
        #   ../pk/pk.c:122: Error: unknown CSR `ssp'
        git checkout 1a52fa4

        # rv32imac
        mkdir build
        pushd build
            ../configure --prefix=`pwd`/install --host=riscv64-unknown-elf --with-arch=rv32imac
            make -j`nproc`
            make install
            cp ./pk $RISCV/riscv64-unknown-elf/bin/pk
        popd

        # rv64imac
        mkdir build64
        pushd build64
            ../configure --prefix=`pwd`/install --host=riscv64-unknown-elf --with-arch=rv64imac
            make -j`nproc`
            make install
            cp ./pk $RISCV/riscv64-unknown-elf/bin/pk64
        popd
    popd
popd

# cleanup
rm -rf /tmp/spike
