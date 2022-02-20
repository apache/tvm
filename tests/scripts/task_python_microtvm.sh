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

set -e
set -u
set -x  # NOTE(areusch): Adding to diagnose flaky timeouts

source tests/scripts/setup-pytest-env.sh

make cython3

# Zephyr
run_pytest ctypes python-microtvm-zephyr-qemu_x86 tests/micro/zephyr --zephyr-board=qemu_x86
run_pytest ctypes python-microtvm-zephyr-qemu_riscv32 tests/micro/zephyr --zephyr-board=qemu_riscv32
run_pytest ctypes python-microtvm-zephyr-qemu_riscv64 tests/micro/zephyr --zephyr-board=qemu_riscv64

# Temporarily removing mps2_an512 from CI due to issue 8728:
# https://github.com/apache/tvm/issues/8728
# run_pytest ctypes python-microtvm-zephyr tests/micro/zephyr --zephyr-board=mps2_an521

# Arduino
run_pytest ctypes python-microtvm-arduino apps/microtvm/arduino/template_project/tests
run_pytest ctypes python-microtvm-arduino-nano33ble tests/micro/arduino  --test-build-only --arduino-board=nano33ble
run_pytest ctypes python-microtvm-arduino-due tests/micro/arduino  --test-build-only --arduino-board=due

# STM32
run_pytest ctypes python-microtvm-stm32 tests/micro/stm32

# Common Tests
run_pytest ctypes python-microtvm-common-qemu_x86 tests/micro/common --board=qemu_x86
run_pytest ctypes python-microtvm-common-due tests/micro/common  --test-build-only --board=due

# # Tutorials running with host CRT
python3 gallery/how_to/work_with_microtvm/micro_tflite.py
python3 gallery/how_to/work_with_microtvm/micro_autotune.py

# Tutorials running with Zephyr
export TVM_MICRO_USE_HW=1
export TVM_MICRO_BOARD=qemu_x86
python3 gallery/how_to/work_with_microtvm/micro_tflite.py
python3 gallery/how_to/work_with_microtvm/micro_autotune.py
