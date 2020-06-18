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

alias make="make -j4"

# Get latest cmake
wget -q https://cmake.org/files/v3.8/cmake-3.8.2-Linux-x86_64.tar.gz
tar xf cmake-3.8.2-Linux-x86_64.tar.gz
export PATH=/cmake-3.8.2-Linux-x86_64/bin/:${PATH}

wget -q https://s3.amazonaws.com/mozilla-games/emscripten/releases/emsdk-portable.tar.gz
tar xf emsdk-portable.tar.gz
cd emsdk-portable
./emsdk update
./emsdk install latest
./emsdk activate latest
# Clone and pull latest sdk
./emsdk install clang-incoming-64bit
./emsdk activate clang-incoming-64bit
cd ..
