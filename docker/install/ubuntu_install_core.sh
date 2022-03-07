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

# install libraries for building c++ core on ubuntu
apt-get update && apt-get install -y --no-install-recommends \
        git make google-mock libgtest-dev cmake wget unzip libtinfo-dev libz-dev \
        libcurl4-openssl-dev libssl-dev libopenblas-dev g++ sudo \
        apt-transport-https graphviz pkg-config curl

if [[ -d /usr/src/googletest ]]; then
  # Single package source (Ubuntu 18.04)
  # googletest is installed via libgtest-dev
  cd /usr/src/googletest && cmake CMakeLists.txt && make && cp -v {googlemock,googlemock/gtest}/*.a /usr/lib
else
  # Split source package (Ubuntu 16.04)
  # libgtest-dev and google-mock
  cd /usr/src/gtest && cmake CMakeLists.txt && make && cp -v *.a /usr/lib
  cd /usr/src/gmock && cmake CMakeLists.txt && make && cp -v *.a /usr/lib
fi
