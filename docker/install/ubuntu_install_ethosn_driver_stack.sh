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

repo_url="https://github.com/Arm-software/ethos-n-driver-stack"
repo_dir="ethosn-driver"
repo_revision="22.08"
install_path="/opt/arm/$repo_dir"

tmpdir=$(mktemp -d)

cleanup()
{
  rm -rf "$tmpdir"
}

trap cleanup 0

# Ubuntu 16.04 dependencies
apt-get update

apt-install-and-clear -y \
    bsdmainutils \
    build-essential \
    cmake \
    cpp \
    git \
    linux-headers-generic \
    python-dev \
    python3 \
    scons \
    wget

cd "$tmpdir"
git clone --branch "$repo_revision" "$repo_url" "$repo_dir"

cd "$repo_dir"/driver
scons install_prefix="$install_path" install
