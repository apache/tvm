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

#install the necessary dependancies, cffi, opencv
wget -q 'https://github.com/siju-samuel/darknet/blob/master/lib/libdarknet.so?raw=true' -O libdarknet.so
debian_version=`cat /etc/debian_version`
if [ "$debian_version" == "stretch/sid" ]; then
    pip2 install opencv-python cffi
fi
pip3 install opencv-python cffi
