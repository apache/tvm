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
    attrs \
    cloudpickle \
    cython \
    decorator \
    mypy \
    numpy~=1.19.5 \
    orderedset \
    packaging \
    Pillow==9.1.0 \
    psutil \
    pytest \
    git+https://github.com/tlc-pack/tlcpack-sphinx-addon.git@7f69989f1c6a6713d0bd7c27f8da2b48344117d3 \
    pytest-profiling \
    pytest-xdist \
    requests \
    scipy \
    Jinja2 \
    synr==0.6.0 \
    junitparser==2.4.2 \
    six \
    tornado \
    pytest-lazy-fixture
