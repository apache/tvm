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

# the minimum cmake is 3.20.0 for LLVM 16+
if [ -z ${1+x} ]; then
    version=3.20.0
else
    version=$1
fi

v=$(echo $version | sed 's/\(.*\)\..*/\1/g')
echo "Installing cmake $version ($v)"
wget https://cmake.org/files/v${v}/cmake-${version}.tar.gz
tar xvf cmake-${version}.tar.gz
cd cmake-${version}
./bootstrap
make -j$(nproc)
make install
cd ..
rm -rf cmake-${version} cmake-${version}.tar.gz
