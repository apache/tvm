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

export PYTHONPATH=python

cp /emsdk-portable/.emscripten ~/.emscripten
source /emsdk-portable/emsdk_env.sh

export EM_CONFIG=${HOME}/.emscripten
export EM_CACHE=${HOME}/.emscripten_cache

echo "Build TVM Web runtime..."
make web

echo "Prepare test libraries..."
python tests/web/prepare_test_libs.py

echo "Start testing..."

for test in tests/web/test_*.js; do
    echo node $test
    node $test
done

echo "All tests finishes..."
