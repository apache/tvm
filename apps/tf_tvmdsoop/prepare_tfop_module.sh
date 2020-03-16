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

TVM_ROOT=$(cd $(dirname $0)/../..; pwd)
echo "TVM_ROOT=${TVM_ROOT}"

export PYTHONPATH=${TVM_ROOT}/python

python -c "import tvm; print(tvm.module.enabled('gpu'))" | grep -e 1
if [[ $? -eq 0 ]]; then 
    echo "Build TF_TVMDSOOP with gpu support"
    CMAKE_OPTIONS="-DUSE_CUDA=ON -DTVM_ROOT=${TVM_ROOT}"
else
    CMAKE_OPTIONS="-DUSE_CUDA=OFF -DTVM_ROOT=${TVM_ROOT}"
fi

mkdir -p build; cd build; cmake .. ${CMAKE_OPTIONS}; make

