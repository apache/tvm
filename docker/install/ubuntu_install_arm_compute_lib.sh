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

repo_url="https://github.com/ARM-software/ComputeLibrary.git"
repo_dir="acl"
install_path="/opt/$repo_dir"
architecture_type=$(uname -i)
target_arch="arm64-v8a" # arm64-v8a / arm64-v8.2-a / armv7a
build_type="native"

tmpdir=$(mktemp -d)

cleanup()
{
  rm -rf "$tmpdir"
}

trap cleanup 0

apt-get update && \
apt-get install -y --no-install-recommends \
    git \
    scons \
    bsdmainutils \
    build-essential

# Install cross-compiler when not building natively.
# Depending on the architecture selected to compile for,
# you may need to install an alternative cross-compiler.
if [ "$architecture_type" != "aarch64" ]; then
  apt-get install -y --no-install-recommends \
    g++-aarch64-linux-gnu \
    gcc-aarch64-linux-gnu
fi

cd "$tmpdir"

git clone "$repo_url" "$repo_dir"

cd "$repo_dir"

# pin version to v20.05
git checkout 6a7771e

if [ "$architecture_type" != "aarch64" ]; then
  build_type="cross_compile"
fi

scons \
  install_dir="$install_path" \
  Werror=1 \
  -j8 \
  debug=0 \
  asserts=0 \
  neon=1 \
  opencl=0 \
  os=linux \
  arch="$target_arch" \
  build="$build_type"
