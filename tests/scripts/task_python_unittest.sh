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

source tests/scripts/setup-pytest-env.sh

# cleanup pycache
find . -type f -path "*.pyc" | xargs rm -f
make cython3

# NOTE: also set by task_python_unittest_gpuonly.sh.
if [ -z "${TVM_UNITTEST_TESTSUITE_NAME:-}" ]; then
    TVM_UNITTEST_TESTSUITE_NAME=python-unittest
fi

# First run minimal test on both ctypes and cython.
run_pytest --parallel ctypes ${TVM_UNITTEST_TESTSUITE_NAME}-platform-minimal-test tests/python/all-platform-minimal-test
run_pytest --parallel cython ${TVM_UNITTEST_TESTSUITE_NAME}-platform-minimal-test tests/python/all-platform-minimal-test

# Then run all unittests on both ctypes and cython.
run_pytest --parallel ctypes ${TVM_UNITTEST_TESTSUITE_NAME} tests/python/unittest
run_pytest --parallel cython ${TVM_UNITTEST_TESTSUITE_NAME} tests/python/unittest
