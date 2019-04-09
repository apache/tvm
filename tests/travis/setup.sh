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

if [ ${TASK} == "python_test" ] || [ ${TASK} == "all_test" ]; then
    if [ ${TRAVIS_OS_NAME} == "osx" ]; then
        brew update
        brew install python3
        python -m pip install --user nose numpy cython
        python3 -m pip install --user nose numpy cython
    fi
fi

if [ ${TASK} == "lint" ] || [ ${TASK} == "all_test" ]; then
    if [ ! ${TRAVIS_OS_NAME} == "osx" ]; then
        pip install --user cpplint 'pylint==1.4.4' 'astroid==1.3.6'
    fi
fi
