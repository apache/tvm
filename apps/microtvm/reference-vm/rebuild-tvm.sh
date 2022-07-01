#!/bin/bash -e
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
#
# "Usage ./apps/microtvm/reference-vm/rebuild-tvm.sh <PLATFORM>"
#

set -e

if [ "$#" -lt 1 -o "$1" == "--help" ]; then
    echo "Usage ./apps/microtvm/reference-vm/rebuild-tvm.sh <PLATFORM>"
    exit -1
fi

platform=$1
shift 1

# Get number of cores for build
if [ -n "${TVM_CI_NUM_CORES}" ]; then
  num_cores=${TVM_CI_NUM_CORES}
else
  # default setup for Vagrantfile
  num_cores=2
fi

cd "$(dirname $0)"
cd "$(git rev-parse --show-toplevel)"
BUILD_DIR="build-microtvm-${platform}"

if [ ! -e "${BUILD_DIR}" ]; then
    mkdir "${BUILD_DIR}"
fi

./tests/scripts/task_config_build_qemu.sh "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake ..
rm -rf standalone_crt host_standalone_crt  # remove stale generated files
make -j${num_cores}
