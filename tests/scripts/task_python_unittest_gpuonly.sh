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

set -euo pipefail

export TVM_TEST_TARGETS="cuda;opencl;metal;rocm;nvptx;opencl -device=mali,aocl_sw_emu"
export PYTEST_ADDOPTS="-m gpu $PYTEST_ADDOPTS"
export TVM_UNITTEST_TESTSUITE_NAME=python-unittest-gpu

./tests/scripts/task_python_unittest.sh

# Kept separate to avoid increasing time needed to run CI, testing
# only minimal functionality of Vulkan runtime.
export TVM_TEST_TARGETS="vulkan -from_device=0"
export PYTEST_ADDOPTS="-m gpu $PYTEST_ADDOPTS"
export TVM_UNITTEST_TESTSUITE_NAME=python-unittest-vulkan

source tests/scripts/setup-pytest-env.sh

run_pytest ctypes ${TVM_UNITTEST_TESTSUITE_NAME} tests/python/unittest/test_target_codegen_vulkan.py
run_pytest cython ${TVM_UNITTEST_TESTSUITE_NAME} tests/python/unittest/test_target_codegen_vulkan.py
