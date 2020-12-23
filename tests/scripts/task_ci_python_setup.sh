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

# Script to setup additional python env.
#
# Use the following command to install the
# package to /workspace/.local, these additional
# packages will have precedence over the system packages.
#
# command: python3 -m pip install --user <package>==<version>
#
echo "Addtiional setup in" ${CI_IMAGE_NAME}

python3 -m pip install --user tlcpack-sphinx-addon==0.1.3 synr==0.2.1
python3 -m pip install --user setuptools_rust tokenizers transformers
