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

# Verilator version
VERILATOR_VERSION="5.002"

# Install dependencies
apt-get update && apt-install-and-clear -y autoconf g++ flex bison

# Install Verilator
git clone --depth 1 --branch v${VERILATOR_VERSION} https://github.com/verilator/verilator
pushd verilator
  autoconf
  ./configure
  make -j$(nproc)
  make install
popd
rm -rf verilator
