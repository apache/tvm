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

# Test frontends that only need CPU resources
set -e
set -u

source tests/scripts/setup-pytest-env.sh
# to avoid openblas threading error
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=1

export TVM_TEST_TARGETS="llvm"

find . -type f -path "*.pyc" | xargs rm -f

# Rebuild cython
make cython3

echo "Running relay TFLite frontend test..."
run_pytest --parallel cython python-frontend-tflite tests/python/frontend/tflite

echo "Running relay Keras frontend test..."
run_pytest --parallel cython python-frontend-keras tests/python/frontend/keras

echo "Running relay Caffe frontend test..."
run_pytest --parallel cython python-frontend-caffe tests/python/frontend/caffe
