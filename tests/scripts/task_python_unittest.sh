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
export PYTEST_ADDOPTS="${CI_PYTEST_ADD_OPTIONS:-} ${PYTEST_ADDOPTS:-}"

# setup tvm-ffi into python folder
uv pip install -v --target=python ./3rdparty/tvm-ffi/

# LLVMModule JIT execution. Installed here as well as in the CI image because a docker/
# change does not reach the test containers in the same run (they use the pinned tag).
# --no-deps keeps the submodule-built tvm-ffi above from being replaced by a PyPI build.
uv pip install --no-deps apache-tvm-ffi-orcjit==0.1.1

python3 -m pytest -vvs -n auto -m "${TVM_TEST_MARKER:-not gpu}" tests/python
