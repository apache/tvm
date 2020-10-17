#!/bin/bash -e
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

TVM_HOME="$1"

# TVM
# NOTE: TVM is presumed to be mounted already by Vagrantfile.
cd "${TVM_HOME}"

tools/microtvm/vm/rebuild-tvm.sh

# NOTE: until the dependencies make it into a top-level pyproject.toml file in main,
# use this approach.
cp tools/microtvm/vm/pyproject.toml .

poetry lock
poetry install
poetry run pip3 install -r ~/zephyr/zephyr/scripts/requirements.txt
