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

function show_usage() {
    cat <<EOF
Usage: docker/install/ubuntu_download_xuantie_qemu.sh <INSTALLATION_PATH>
INSTALLATION_PATH is the installation path for the tool.
EOF
}

if [ "$#" -lt 1 -o "$1" == "--help" -o "$1" == "-h" ]; then
    show_usage
    exit 1
fi

INSTALLATION_PATH=$1

# Create installation path directory
mkdir -p "${INSTALLATION_PATH}"

QEMU_DATE="20220623-0307"
QEMU_SOURCE_ID="1655972947885"
QEMU_ARCH="x86_64-Ubuntu-18.04"
QEMU_BASE="xuantie-qemu-${QEMU_ARCH}-${QEMU_DATE}"
QEMU_EXT="tar.gz"
QEMU_URL="https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//${QEMU_SOURCE_ID}/${QEMU_BASE}.${QEMU_EXT}"
DOWNLOAD_PATH="/tmp/${QEMU_BASE}.${QEMU_EXT}"

wget ${QEMU_URL} -O "${DOWNLOAD_PATH}"
tar -xf "${DOWNLOAD_PATH}" -C "${INSTALLATION_PATH}" --strip-components=1
rm $DOWNLOAD_PATH

# Remove non riscv64 binaries? (TODO)
# ls $INSTALLATION_PATH/bin | grep -v qemu-riscv64 | xargs -i rm -rf $INSTALLATION_PATH/bin/{}
# ls $INSTALLATION_PATH | grep -v bin | xargs -i rm -rf $INSTALLATION_PATH/{}

echo "SUCCESS"
