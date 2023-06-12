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

source tests/scripts/setup-pytest-env.sh
# to avoid CI thread throttling.
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=1

export PYTHONPATH=${PYTHONPATH}:${TVM_PATH}/vta/python
export VTA_HW_PATH=`pwd`/3rdparty/vta-hw

# disable fsim test for now
exit 0

# cleanup pycache
find . -type f -path "*.pyc" | xargs rm -f

rm -rf ~/.tvm

# Rebuild cython
make cython3

# Reset default fsim simulation
cp ${VTA_HW_PATH}/config/fsim_sample.json ${VTA_HW_PATH}/config/vta_config.json

# Run unit tests in functional/fast simulator
echo "Running unittest in fsim..."
run_pytest cython python-vta-fsim-unittest ${TVM_PATH}/vta/tests/python/unittest

# Run unit tests in functional/fast simulator
echo "Running integration test in fsim..."
run_pytest cython python-vta-fsim-integration ${TVM_PATH}/vta/tests/python/integration
