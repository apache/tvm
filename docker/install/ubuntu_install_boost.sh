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

set -euxo pipefail

cleanup() {
  rm -rf /boost_1_67_0 /boost_1_67_0.tar.gz
}

trap cleanup 0

# NOTE: by default, tvm-venv python is used. Install boost on the system.
PATH=${PATH/${TVM_VENV}\/bin:/}

curl -LO https://archives.boost.io/release/1.67.0/source/boost_1_67_0.tar.gz
BOOST_HASH=8c247e040303a97895cee9c9407ef205e2c3ab09f0b8320997835ad6221dff23a87231629498ccfd0acca473f74e9ec27b8bd774707b062228df1e5f72d44c92
echo "$BOOST_HASH" boost_1_67_0.tar.gz | sha512sum -c
tar -xf boost_1_67_0.tar.gz

pushd boost_1_67_0
./bootstrap.sh --with-python="$(which python3.8)"
./b2 install --with-python --with-system --with-filesystem --with-thread --with-regex
popd

ln -s /usr/local/lib/libboost_python38.so.1.67.0 /usr/local/lib/libboost_python.so
ln -s /usr/local/lib/libboost_python38.a /usr/local/lib/libboost_python.a
