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
# to avoid openblas threading error
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=1

export TVM_TEST_TARGETS="llvm;cuda"

find . -type f -path "*.pyc" | xargs rm -f

# Rebuild cython
make cython3

# Enable tvm.testing decorators in the ONNX importer test (not enabling in the other tests because we
# they do not consistently use the decorators to indicate that tests should run on GPU)
# In the future, we should enable tvm.testing decorators for all the test files.

echo "Running relay MXNet frontend test..."
TVM_PYTHON_FFI_TYPES=cython run_pytest python-frontend-mxnet tests/python/frontend/mxnet

echo "Running relay ONNX frontend test..."
PYTEST_ADDOPTS="-m gpu $PYTEST_ADDOPTS" run_pytest cython python-frontend-onnx tests/python/frontend/onnx

echo "Running relay CoreML frontend test..."
TVM_PYTHON_FFI_TYPES=cython run_pytest python-frontend-coreml tests/python/frontend/coreml

echo "Running relay Tensorflow frontend test..."
TVM_PYTHON_FFI_TYPES=cython run_pytest python-frontend-tensorflow tests/python/frontend/tensorflow

echo "Running relay caffe2 frontend test..."
TVM_PYTHON_FFI_TYPES=cython run_pytest python-frontend-caffe2 tests/python/frontend/caffe2

echo "Running relay DarkNet frontend test..."
TVM_PYTHON_FFI_TYPES=cython run_pytest python-frontend-darknet tests/python/frontend/darknet

echo "Running relay PyTorch frontend test..."
TVM_PYTHON_FFI_TYPES=cython run_pytest python-frontend-pytorch tests/python/frontend/pytorch
