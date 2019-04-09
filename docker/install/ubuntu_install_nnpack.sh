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

apt-get update && apt-get install -y --no-install-recommends git cmake

# TODO: specific tag?
git clone https://github.com/Maratyszcza/NNPACK NNPACK
(cd NNPACK && git checkout 1e005b0c2)

mkdir -p NNPACK/build
cd NNPACK/build
cmake -DCMAKE_INSTALL_PREFIX:PATH=. -DNNPACK_INFERENCE_ONLY=OFF -DNNPACK_CONVOLUTION_ONLY=OFF -DNNPACK_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && make -j4 && make install
cd -
