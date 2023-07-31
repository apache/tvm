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
pip3 install --upgrade \
    "Pygments>=2.4.0" \
    attrs \
    cloudpickle \
    cython \
    decorator \
    mypy \
    numpy==1.21.* \
    orderedset \
    packaging \
    Pillow==9.1.0 \
    psutil \
    pytest \
    git+https://github.com/tlc-pack/tlcpack-sphinx-addon.git@768ec1dce349fe4708f6ad68be1ebb3f3dabafa1 \
    pytest-profiling \
    pytest-xdist \
    pytest-rerunfailures==10.2 \
    requests \
    Jinja2 \
    junitparser==2.4.2 \
    six \
    tornado \
    pytest-lazy-fixture \
    git+https://github.com/jax-ml/ml_dtypes.git@v0.2.0
