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

export PYTHONPATH="$(pwd)/python"
export PYTEST_ADDOPTS="-s -vv ${CI_PYTEST_ADD_OPTIONS:-} ${PYTEST_ADDOPTS:-}"
mkdir -p build/pytest-results

# cleanup pycache
find . -type f -path "*.pyc" | xargs rm -f

# setup tvm-ffi into python folder
uv pip install -v --target=python ./3rdparty/tvm-ffi/

# NOTE: also set by task_python_unittest_gpuonly.sh.
if [ -z "${TVM_UNITTEST_TESTSUITE_NAME:-}" ]; then
    TVM_UNITTEST_TESTSUITE_NAME=python-unittest
fi
PYTEST_SHARD_SUFFIX="${TVM_SHARD_INDEX:+-shard-${TVM_SHARD_INDEX}}"

# First run the minimal platform test.  A GPU-only run can select no tests here.
python3 -m pytest -n auto \
    -o "junit_suite_name=${TVM_UNITTEST_TESTSUITE_NAME}-platform-minimal-test${PYTEST_SHARD_SUFFIX}" \
    "--junit-xml=build/pytest-results/${TVM_UNITTEST_TESTSUITE_NAME}-platform-minimal-test${PYTEST_SHARD_SUFFIX}.xml" \
    --junit-prefix=cython \
    tests/python/all-platform-minimal-test || [ "$?" -eq 5 ]

# Then run all unit tests.
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
  "s_tir/transform"
  "tirx-analysis"
  "tirx-base"
  "tirx-transform"
  "tirx"
  "tvmscript"
  "relax"
)

PYTEST_TARGETS=()
for TEST_FILE in "${TEST_FILES[@]}"; do
    PYTEST_TARGETS+=("tests/python/${TEST_FILE}")
done

python3 -m pytest -n auto --dist=loadgroup \
    -o "junit_suite_name=${TVM_UNITTEST_TESTSUITE_NAME}${PYTEST_SHARD_SUFFIX}" \
    "--junit-xml=build/pytest-results/${TVM_UNITTEST_TESTSUITE_NAME}${PYTEST_SHARD_SUFFIX}.xml" \
    --junit-prefix=cython \
    "${PYTEST_TARGETS[@]}"
