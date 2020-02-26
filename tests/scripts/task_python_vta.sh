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

export PYTHONPATH=python:vta/python:topi/python

# cleanup pycache
find . -type f -path "*.pyc" | xargs rm -f

rm -rf ~/.tvm

# Rebuild cython
make cython3

# Reset default fsim simulation
cp vta/config/fsim_sample.json vta/config/vta_config.json

# Run unit tests in functional/fast simulator
echo "Running unittest in fsim..."
python3 -m pytest -v vta/tests/python/unittest

# Run unit tests in functional/fast simulator
echo "Running integration test in fsim..."
python3 -m pytest -v vta/tests/python/integration
