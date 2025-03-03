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

CMAKE_VERSION="3.30.4"
CMAKE_SHA256="c759c97274f1e7aaaafcb1f0d261f9de9bf3a5d6ecb7e2df616324a46fe704b2"

# parse argument
CMAKE_VERSION=${1:-$CMAKE_VERSION}
CMAKE_SHA256=${2:-$CMAKE_SHA256}

v=$(echo $CMAKE_VERSION | sed 's/\(.*\)\..*/\1/g')
echo "Installing cmake $CMAKE_VERSION ($v)"
wget https://cmake.org/files/v${v}/cmake-${CMAKE_VERSION}.tar.gz
echo "$CMAKE_SHA256" cmake-${CMAKE_VERSION}.tar.gz | sha256sum -c
tar xvf cmake-${CMAKE_VERSION}.tar.gz
pushd cmake-${CMAKE_VERSION}
  ./bootstrap
  make -j$(nproc)
  make install
popd
rm -rf cmake-${CMAKE_VERSION} cmake-${CMAKE_VERSION}.tar.gz
