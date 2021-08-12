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
set -x  # NOTE(areusch): Adding to diagnose flaky timeouts

source tests/scripts/setup-pytest-env.sh

make cython3
run_pytest ctypes python-microtvm-zephyr tests/micro/zephyr --microtvm-platforms=qemu_x86
run_pytest ctypes python-microtvm-zephyr tests/micro/zephyr --microtvm-platforms=mps2_an521
run_pytest ctypes python-microtvm-arduino-due tests/micro/arduino  --test-build-only --microtvm-platforms=due
run_pytest ctypes python-microtvm-arduino-due tests/micro/arduino  --test-build-only --microtvm-platforms=nano33ble
