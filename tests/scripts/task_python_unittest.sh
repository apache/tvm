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

# cleanup pycache
find . -type f -path "*.pyc" | xargs rm -f

# setup tvm-ffi into python folder
uv pip install -v --target=python ./3rdparty/tvm-ffi/

# First run the minimal platform test.  A GPU-only run can select no tests here.
if [ ! -d tests/python/all-platform-minimal-test ]; then
    echo "Missing pytest target: tests/python/all-platform-minimal-test" >&2
    exit 1
fi
python3 -m pytest -n auto tests/python/all-platform-minimal-test || [ "$?" -eq 5 ]

# Then run all unit tests.
TEST_FILES=(
  "arith"
  "ci"
  "codegen"
  "driver"
  "ir"
  "runtime"
  "target"
  "te"
  "testing"
  "s_tir/base"
  "s_tir/schedule"
  "s_tir/dlight"
  "s_tir/analysis"
  "s_tir/meta_schedule"
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
    TEST_PATH="tests/python/${TEST_FILE}"
    if [ ! -e "${TEST_PATH}" ]; then
        echo "Missing pytest target: ${TEST_PATH}" >&2
        exit 1
    fi
    PYTEST_TARGETS+=("${TEST_PATH}")
done

# Do not mask pytest's exit 5: an unexpectedly empty broad suite must fail CI.
python3 -m pytest -n auto --dist=loadgroup "${PYTEST_TARGETS[@]}"
