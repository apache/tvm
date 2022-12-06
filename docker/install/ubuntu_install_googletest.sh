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

if [ $# -eq 0 ]; then
    tmpdir=$(mktemp -d)

    cleanup()
    {
      rm -rf "$tmpdir"
    }

    trap cleanup 0
else
    tmpdir=$1
    mkdir -p "$tmpdir"
fi

# GoogleTest uses a Live-at-Head philosophy:
# https://github.com/google/googletest#live-at-head
# therefore we need to grab a specific hash and update it
# periodically to match the head of the repo
repo_url="https://github.com/google/googletest"
repo_revision="830fb567285c63ab5b5873e2e8b02f2249864916"

archive_name="${repo_revision}.tar.gz"
archive_url="${repo_url}/archive/${archive_name}"
archive_hash="10f10ed771efc64a1d8234a7e4801838a468f8990e5d6d8fcf63e89f8d1455c4f9c5adc0bb829669f381609a9abf84e4c91a7fdd7404630f375f38fb485ef0eb"

cd "$tmpdir"

curl -sL "${archive_url}" -o "${archive_name}"
echo "$archive_hash" ${archive_name} | sha512sum -c
tar xf "${archive_name}" --strip-components=1

mkdir build
cd build

# CMake doesn't search /usr/local/lib/<arch> properly for GoogleTest
# so we use /usr/lib where it does search
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib ..
cmake --build . --target install
