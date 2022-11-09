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

if [ $# -gt 0 ]; then
    BUILD_DIR="$1"
elif [ -n "${TVM_BUILD_PATH:-}" ]; then
    # TVM_BUILD_PATH may contain multiple space-separated paths.  If
    # so, use the first one.
    BUILD_DIR=$(IFS=" "; set -- $TVM_BUILD_PATH; echo $1)
else
    BUILD_DIR=build
fi

# Python is required by apps/bundle_deploy
source tests/scripts/setup-pytest-env.sh

export LD_LIBRARY_PATH="lib:${LD_LIBRARY_PATH:-}"
# NOTE: important to use abspath, when VTA is enabled.
export VTA_HW_PATH=`pwd`/3rdparty/vta-hw

# to avoid CI thread throttling.
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=1

# Build cpptest suite
python3 tests/scripts/task_build.py \
    --sccache-bucket tvm-sccache-prod \
    --cmake-target cpptest \
    --build-dir "${BUILD_DIR}"

# crttest requries USE_MICRO to be enabled.
if grep -Fq "USE_MICRO ON" ${BUILD_DIR}/TVMBuildOptions.txt; then
  pushd "${BUILD_DIR}"
  ninja crttest
  popd
fi

pushd "${BUILD_DIR}"
ctest --gtest_death_test_style=threadsafe
popd

# Test MISRA-C runtime. It requires USE_MICRO to be enabled.
if grep -Fq "USE_MICRO ON" ${BUILD_DIR}/TVMBuildOptions.txt; then
  pushd apps/bundle_deploy
  rm -rf build
  make test_dynamic test_static
  popd
fi
