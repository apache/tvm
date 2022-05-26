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
apt-install-and-clear -y software-properties-common python3.7-dev python3-setuptools python3.7-venv

python3 -mvenv /opt/tvm-venv

# Pin pip and setuptools versions
/opt/tvm-venv/bin/pip3 install pip==19.3.1 setuptools==58.4.0
