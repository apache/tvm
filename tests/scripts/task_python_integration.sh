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
export PYTHONPATH=${PYTHONPATH}:${TVM_PATH}/apps/extension/python
export LD_LIBRARY_PATH="build:${LD_LIBRARY_PATH:-}"

# to avoid CI CPU thread throttling.
export TVM_BIND_THREADS=0
export TVM_NUM_THREADS=2

# NOTE: also set by task_python_integration_gpuonly.sh.
if [ -z "${TVM_INTEGRATION_TESTSUITE_NAME:-}" ]; then
    TVM_INTEGRATION_TESTSUITE_NAME=python-integration
fi

# cleanup pycache
find . -type f -path "*.pyc" | xargs rm -f

# Test TVM
make cython3

# Test extern package
cd apps/extension
rm -rf lib
make
cd ../..

run_pytest ctypes ${TVM_INTEGRATION_TESTSUITE_NAME}-extensions-0 apps/extension/tests
run_pytest cython ${TVM_INTEGRATION_TESTSUITE_NAME}-extensions-1 apps/extension/tests

# Test dso plugin
cd apps/dso_plugin_module
rm -rf lib
make
cd ../..
run_pytest ctypes ${TVM_INTEGRATION_TESTSUITE_NAME}-dso_plugin_module-0 apps/dso_plugin_module
run_pytest cython ${TVM_INTEGRATION_TESTSUITE_NAME}-dso_plugin_module-1 apps/dso_plugin_module

# Do not enable TensorFlow op
# TVM_FFI=cython sh prepare_and_test_tfop_module.sh
# TVM_FFI=ctypes sh prepare_and_test_tfop_module.sh

run_pytest ctypes ${TVM_INTEGRATION_TESTSUITE_NAME}-integration tests/python/integration

# Ignoring Arm(R) Ethos(TM)-U NPU tests in the collective to run to run them in parallel in the next step.
run_pytest ctypes ${TVM_INTEGRATION_TESTSUITE_NAME}-contrib tests/python/contrib --ignore=tests/python/contrib/test_ethosu --ignore=tests/python/contrib/test_cmsisnn
# forked is needed because the global registry gets contaminated
TVM_TEST_TARGETS="${TVM_RELAY_TEST_TARGETS:-llvm;cuda}" \
    run_pytest ctypes ${TVM_INTEGRATION_TESTSUITE_NAME}-relay tests/python/relay --ignore=tests/python/relay/aot

# OpenCL texture test. Deselected specific tests that fails  in CI
TVM_TEST_TARGETS="${TVM_RELAY_OPENCL_TEXTURE_TARGETS:-opencl}" \
    run_pytest ctypes ${TVM_INTEGRATION_TESTSUITE_NAME}-opencl-texture tests/python/relay/opencl_texture
# Command line driver test
run_pytest ctypes ${TVM_INTEGRATION_TESTSUITE_NAME}-driver tests/python/driver

# Target test
run_pytest ctypes ${TVM_INTEGRATION_TESTSUITE_NAME}-target tests/python/target

# Do not enable OpenGL
# run_pytest ctypes ${TVM_INTEGRATION_TESTSUITE_NAME}-webgl tests/webgl
