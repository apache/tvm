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

# This script builds TVM with default settings inside the ci-gpu CI image, then
# builds the docs.

set -eux

IMAGE_ID=$(sudo docker run -d -v "$(pwd)":/tvm -it tlcpack/ci-gpu:v0.78)

trap 'sudo docker stop "$IMAGE_ID"' EXIT

docker_run() {
    sudo docker exec -w "$1" \
        -e TVM_TUTORIAL_EXEC_PATTERN=none \
        -e CI_IMAGE_NAME=ci-gpu \
        -i "$IMAGE_ID" bash -c "$2"
}

docker_run /tvm 'mkdir -p build'
docker_run / 'python3 -m pip install --user tlcpack-sphinx-addon==0.2.1 synr==0.5.0'
docker_run /tvm/build 'cmake ..'
docker_run /tvm/build 'cmake --build . -- -j $(nproc)'
docker_run /tvm/docs 'make html'

