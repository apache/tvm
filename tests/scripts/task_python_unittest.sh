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
make cython3

# NOTE: also set by task_python_unittest_gpuonly.sh.
if [ -z "${TVM_UNITTEST_TESTSUITE_NAME:-}" ]; then
    TVM_UNITTEST_TESTSUITE_NAME=python-unittest
fi

# First run minimal test on both ctypes and cython.
run_pytest ctypes ${TVM_UNITTEST_TESTSUITE_NAME}-platform-minimal-test-0 tests/python/all-platform-minimal-test
run_pytest cython ${TVM_UNITTEST_TESTSUITE_NAME}-platform-minimal-test-1 tests/python/all-platform-minimal-test

# Then run all unittests on both ctypes and cython.
TEST_FILES=(
  "arith"
  "auto_scheduler"
  "autotvm"
  "codegen"
  "ir"
  "meta_schedule"
  "micro"
  "runtime"
  "te"
  "testing"
  "tir-analysis"
  "tir-base"
  "tir-schedule"
  "tir-transform"
  "tir-usmp"
  "tvmscript"
  "usmp"
)

for TEST_FILE in ${TEST_FILES[@]}; do
    run_pytest cython ${TEST_FILE}-1, tests/python/${TEST_FILE}
done

# Then run CI tests
run_pytest ctypes ${TVM_UNITTEST_TESTSUITE_NAME}-ci tests/python/ci
