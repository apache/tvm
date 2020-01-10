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

apt-get install -y --no-install-recommends make bison flex
wget -q ftp://icarus.com/pub/eda/verilog/v10/verilog-10.1.tar.gz
tar xf verilog-10.1.tar.gz
cd verilog-10.1
./configure --prefix=/usr
make install -j8
cd ..
rm -rf verilog-10.1 verilog-10.1.tar.gz
