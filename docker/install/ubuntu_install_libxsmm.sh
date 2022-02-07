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

pushd /usr/local/
wget -q https://github.com/libxsmm/libxsmm/archive/refs/tags/1.17.tar.gz
tar -xzf 1.17.tar.gz
pushd ./libxsmm-1.17/
make STATIC=0 -j10
mv libxsmm-1.17/include/* /usr/local/include/
mv libxsmm-1.17/lib/*so /usr/local/lib/
rm -rf 1.17.tar.gz libxsmm-1.17
popd
popd
