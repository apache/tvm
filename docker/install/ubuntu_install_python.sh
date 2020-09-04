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

# install python and pip, don't modify this, modify install_python_package.sh
apt-get update
apt-get install -y python-dev

# python 3.6
apt-get install -y software-properties-common

add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python-pip python-dev python3.6 python3.6-dev

rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3

# Install pip
cd /tmp && wget -q https://bootstrap.pypa.io/get-pip.py && python2 get-pip.py && python3.6 get-pip.py

# Pin pip version
pip3 install pip==19.3.1
