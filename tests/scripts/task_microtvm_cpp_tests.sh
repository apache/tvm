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

BUILD_DIR=$1

# Python is required by apps/bundle_deploy
source tests/scripts/setup-pytest-env.sh

export LD_LIBRARY_PATH="lib:${LD_LIBRARY_PATH:-}"
# NOTE: important to use abspath, when VTA is enabled.
VTA_HW_PATH=$(pwd)/3rdparty/vta-hw
export VTA_HW_PATH

# to avoid CI thread throttling.
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=1

# crttest requries USE_MICRO to be enabled.
./build/crttest

# Test MISRA-C runtime. It requires USE_MICRO to be enabled.
pushd apps/bundle_deploy
rm -rf build
make test_dynamic test_static
popd
