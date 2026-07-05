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

# setup tvm-ffi into python folder
uv pip install -v --target=python ./3rdparty/tvm-ffi/

# cleanup pycache
find . -type f -path "*.pyc" | xargs rm -f

python3 -m pytest -n auto \
    -o junit_suite_name=python-topi-nightly \
    --junit-xml=build/pytest-results/python-topi-nightly.xml \
    --junit-prefix=cython \
    tests/python/topi/nightly

# Tensorflow device verification and network tests on nightly
export CI_ENV_NIGHTLY
python3 tests/python/relax/test_frontend_tflite.py
