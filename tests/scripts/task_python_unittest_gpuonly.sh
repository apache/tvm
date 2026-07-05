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

# Every GPU test carries the `gpu` marker; the specific backend is gated by skipif.
export PYTEST_ADDOPTS="-m gpu ${PYTEST_ADDOPTS:-}"

# Test most of the enabled runtimes here.
# TODO: disabled opencl tests due to segmentation fault.
export TVM_TEST_TARGETS='cuda;metal;rocm;nvptx'

./tests/scripts/task_python_unittest.sh

# Kept separate to avoid increasing time needed to run CI, testing
# only minimal functionality of Vulkan runtime.
export TVM_TEST_TARGETS='{"kind":"vulkan","from_device":0}'

export PYTHONPATH="$(pwd)/python"
export PYTEST_ADDOPTS="${CI_PYTEST_ADD_OPTIONS:-} ${PYTEST_ADDOPTS:-}"

python3 -m pytest -vvs -n auto tests/python/codegen/test_target_codegen_vulkan.py
