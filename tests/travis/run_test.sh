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

if [ ${TASK} == "lint" ] || [ ${TASK} == "all_test" ]; then
    if [ ! ${TRAVIS_OS_NAME} == "osx" ]; then
        ./tests/scripts/task_lint.sh || exit -1
    fi
fi

cp make/config.mk config.mk
echo "USE_CUDA=0" >> config.mk
echo "USE_RPC=1" >> config.mk

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    echo "USE_OPENCL=1" >> config.mk
    echo "USE_METAL=1" >> config.mk
else
    # use g++-4.8 for linux
    if [ ${CXX} == "g++" ]; then
        export CXX=g++-4.8
    fi
    echo "USE_OPENCL=0" >> config.mk
fi

if [ ${TASK} == "verilog_test" ] || [ ${TASK} == "all_test" ]; then
    if [ ! ${TRAVIS_OS_NAME} == "osx" ]; then
        make -f tests/scripts/packages.mk iverilog
        make all || exit -1
        ./tests/scripts/task_verilog_test.sh || exit -1
    fi
fi

if [ ${TASK} == "cpp_test" ] || [ ${TASK} == "all_test" ]; then
    make -f dmlc-core/scripts/packages.mk gtest
    ./tests/scripts/task_cpp_unittest.sh || exit -1
fi

if [ ${TASK} == "python_test" ] || [ ${TASK} == "all_test" ]; then
    make all || exit -1
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        ./tests/scripts/task_python_unittest.sh || exit -1
    else
        nosetests -v tests/python/unittest || exit -1
        nosetests3 -v tests/python/unittest || exit -1
    fi
fi
