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

# Install the AArch64 Architecture Envelope Model (AEM)

set -e
set -u
set -o pipefail

tmpdir=$(mktemp -d)

cleanup()
{
    rm -rf "$tmpdir"
}

trap cleanup 0

pushd "$tmpdir"

# Install GCC toolchain
gcc_install_dir="/opt/arm/gcc-aarch64-none-elf"
gcc_url="https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-elf.tar.xz?rev=28d5199f6db34e5980aae1062e5a6703&hash=D87D4B558F0A2247B255BA15C32A94A9F354E6A8"
gcc_sha="7fe7b8548258f079d6ce9be9144d2a10bd2bf93b551dafbf20fe7f2e44e014b8"
gcc_tar="arm-gnu-toolchain-13.2.rel1-x86_64-aarch64-none-linux-gnu.tar.xz"
mkdir -p $gcc_install_dir
curl --retry 64 -sSL $gcc_url -o $gcc_tar
echo "$gcc_sha $gcc_tar" | sha256sum --check
tar -xf $gcc_tar -C $gcc_install_dir --strip-components=1

# Download FVP
fvp_dir="/opt/arm/fvp"
fvp_url="https://developer.arm.com/-/media/Files/downloads/ecosystem-models/FVP_Base_RevC-2xAEMvA_11.24_11_Linux64.tgz"
fvp_sha="0f132334834cbc66889a62dd72057c976d7c7dfcfeec21799e9c78fb2ce24720"
curl --retry 64 -sSL $fvp_url -o fvp.tgz
echo "$fvp_sha fvp.tgz" | sha256sum --check
mkdir -p "$fvp_dir"
tar -xzf fvp.tgz -C "$fvp_dir"
rm -rf doc  # Remove some documentation bundled with the package
