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

# cleanup pycache
find . -type f -path "*.pyc" | xargs rm -f

# setup tvm-ffi into python folder
uv pip install -v --target=python ./3rdparty/tvm-ffi/

# NOTE: also set by task_python_unittest_gpuonly.sh.
if [ -z "${TVM_UNITTEST_TESTSUITE_NAME:-}" ]; then
    TVM_UNITTEST_TESTSUITE_NAME=python-unittest
fi

# First run minimal test on both ctypes and cython.
run_pytest ${TVM_UNITTEST_TESTSUITE_NAME}-platform-minimal-test tests/python/all-platform-minimal-test

# Then run all unittests on both ctypes and cython.
TEST_FILES=(
  "ffi"
  "arith"
  "ci"
  "codegen"
  "driver"
  "ir"
  "meta_schedule"
  "runtime"
  "target"
  "te"
  "testing"
  "s_tir/base"
  "s_tir/schedule"
  "s_tir/dlight"
  "s_tir/analysis"
  "tirx-analysis"
  "tirx-base"
  "tirx-transform"
  "tvmscript"
  "relax"
)

for TEST_FILE in ${TEST_FILES[@]}; do
    run_pytest ${TEST_FILE}, tests/python/${TEST_FILE}
done

# Run CPU-safe TIRX DSL tests that live under tests/python/tirx.  The full
# directory also contains CUDA/codegen/operator tests, so keep those in the
# GPU/codegen-specific suites instead of adding them to the general Python
# unittest job.
run_pytest tirx-dsl tests/python/tirx \
    --ignore=tests/python/tirx/codegen \
    --ignore=tests/python/tirx/operator \
    --ignore=tests/python/tirx/test_buffer_print.py \
    --ignore=tests/python/tirx/test_control_flow.py

run_pytest tirx-operator-trn \
    tests/python/tirx/operator/tile_primitive/test_dispatcher.py \
    tests/python/tirx/operator/tile_primitive/trn

run_pytest tirx-codegen-nki tests/python/tirx/codegen/test_codegen_nki.py
