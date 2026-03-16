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
set -o pipefail

# install libraries for python package on ubuntu
uv pip install --upgrade \
    "Pygments~=2.19" \
    "cloudpickle~=3.1" \
    "cython~=3.0" \
    "mypy~=1.15" \
    numpy==1.26.* \
    "orderedset~=2.0" \
    "packaging~=25.0" \
    Pillow==12.1.1 \
    "psutil~=7.0" \
    "pytest~=8.3" \
    git+https://github.com/tlc-pack/tlcpack-sphinx-addon.git@768ec1dce349fe4708f6ad68be1ebb3f3dabafa1 \
    "pytest-profiling~=1.8" \
    "pytest-xdist~=3.6" \
    pytest-rerunfailures==16.1 \
    "requests~=2.32" \
    "scipy~=1.13" \
    "Jinja2~=3.1" \
    junitparser==4.0.2 \
    "six~=1.17" \
    "tornado~=6.4" \
    "ml_dtypes~=0.5"
