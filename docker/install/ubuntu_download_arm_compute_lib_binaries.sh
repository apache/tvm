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
architecture_type=$(uname -i)
# Install cross-compiler when not building natively.
# Depending on the architecture selected to compile for,
# you may need to install an alternative cross-compiler.
if [ "$architecture_type" != "aarch64" ]; then
  apt-get update
  apt-install-and-clear -y --no-install-recommends \
    g++-aarch64-linux-gnu \
    gcc-aarch64-linux-gnu
fi

compute_lib_version="v23.08"
compute_lib_variant="arm64-v8a-neon"
compute_lib_full_name="arm_compute-${compute_lib_version}-bin-linux-${compute_lib_variant}"
compute_lib_base_url="https://github.com/ARM-software/ComputeLibrary/releases/download/${compute_lib_version}"
compute_lib_file_name="${compute_lib_full_name}.tar.gz"
compute_lib_download_url="${compute_lib_base_url}/${compute_lib_file_name}"

target_lib="${compute_lib_variant}"

# uncomment line below if you need asserts/debug version of the library
# target_lib="${target_lib}-asserts"

extract_dir="${compute_lib_full_name}"
install_path="/opt/acl"

tmpdir=$(mktemp -d)

cleanup()
{
  rm -rf "$tmpdir"
}

trap cleanup 0

cd "$tmpdir"

curl -sL "${compute_lib_download_url}" -o "${compute_lib_file_name}"
tar xzf "${compute_lib_file_name}"

rm -rf "${install_path}"
mkdir -p "${install_path}"
cp -r "${extract_dir}/include" "${install_path}/"
cp -r "${extract_dir}/arm_compute" "${install_path}/include/"
cp -r "${extract_dir}/support" "${install_path}/include/"
cp -r "${extract_dir}/utils" "${install_path}/include/"
cp -r "${extract_dir}/lib/${target_lib}" "${install_path}/lib"
