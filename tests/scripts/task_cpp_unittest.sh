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

# Python is required by apps/bundle_deploy
source tests/scripts/setup-pytest-env.sh

export LD_LIBRARY_PATH="lib:${LD_LIBRARY_PATH:-}"
# NOTE: important to use abspath, when VTA is enabled.
export VTA_HW_PATH=`pwd`/3rdparty/vta-hw

# to avoid CI thread throttling.
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=1

# Build cpptest suite
make cpptest -j2

# "make crttest" requires USE_MICRO to be enabled, which is not always the case.
if grep crttest build/Makefile > /dev/null; then
    make crttest  # NOTE: don't parallelize, due to issue with build deps.
fi

cd build && ctest --gtest_death_test_style=threadsafe && cd ..

# Test MISRA-C runtime
cd apps/bundle_deploy
rm -rf build
make test_dynamic test_static
cd ../..

# Test Arm(R) Cortex(R)-M55 CPU and Ethos(TM)-U55 NPU demo app
FVP_PATH="/opt/arm/FVP_Corstone_SSE-300_Ethos-U55"
VELA_INSTALLED=$(pip3 list | grep vela)
if [ -d $FVP_PATH ] && [ -n "$VELA_INSTALLED" ]; then
    sudo pip3 install -e python
    cd apps/microtvm/ethosu
    ./run_demo.sh --fvp_path $FVP_PATH --cmake_path /opt/arm/cmake/bin/cmake
    cd ../../..
fi
