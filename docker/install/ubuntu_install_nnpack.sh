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

apt-get update && apt-install-and-clear -y --no-install-recommends git cmake python-setuptools

git clone https://github.com/Maratyszcza/NNPACK NNPACK
git clone https://github.com/Maratyszcza/pthreadpool  NNPACK/pthreadpool

# Use specific versioning tag.
(cd NNPACK && git checkout 70a77f485)
(cd NNPACK/pthreadpool && git checkout 43edadc)

mkdir -p NNPACK/build
cd NNPACK/build
cmake -DCMAKE_INSTALL_PREFIX:PATH=. -DNNPACK_INFERENCE_ONLY=OFF -DNNPACK_CONVOLUTION_ONLY=OFF -DNNPACK_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DPTHREADPOOL_SOURCE_DIR=pthreadpool ..
make -j2
make install
cd -
