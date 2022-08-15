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

pushd apps/microtvm/cmsisnn
 timeout 5m ./run_demo.sh
popd

pushd apps/microtvm/zephyr_cmsisnn
 timeout 5m ./run_demo.sh
popd

pushd apps/microtvm/ethosu
FVP_PATH="/opt/arm/FVP_Corstone_SSE-300_Ethos-U55"
CMAKE_PATH="/opt/arm/cmake/bin/cmake"
FREERTOS_PATH="/opt/freertos/FreeRTOSv202112.00"

 timeout 5m ./run_demo.sh --fvp_path $FVP_PATH --cmake_path $CMAKE_PATH
 timeout 5m ./run_demo.sh --fvp_path $FVP_PATH --cmake_path $CMAKE_PATH --freertos_path $FREERTOS_PATH
popd
