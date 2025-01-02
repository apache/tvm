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
#

#
# Install Zephyr SDK
#
# Usage: docker/install/ubuntu_install_zephyr_sdk.sh <INSTALLATION_PATH>
# INSTALLATION_PATH is the installation path for the SDK.
#

set -e
set -x

function show_usage() {
    cat <<EOF
Usage: docker/install/ubuntu_install_zephyr_sdk.sh <INSTALLATION_PATH>
INSTALLATION_PATH is the installation path for the SDK.
EOF
}

if [ "$#" -lt 1 -o "$1" == "--help" -o "$1" == "-h" ]; then
    show_usage
    exit -1
fi

INSTALLATION_PATH=$1
shift

ZEPHYR_SDK_VERSION=0.16.9
ZEPHYR_SDK_FILE_SHA=9b4b96f8df594801a84fce1aa112a84cdfcd430efc2a74229907b6421fb859a8
ZEPHYR_SDK_FILE_NAME=zephyr-sdk-${ZEPHYR_SDK_VERSION}_linux-x86_64.tar.xz
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v${ZEPHYR_SDK_VERSION}/${ZEPHYR_SDK_FILE_NAME}
echo "$ZEPHYR_SDK_FILE_SHA ${ZEPHYR_SDK_FILE_NAME}" | sha256sum --check

mkdir ${INSTALLATION_PATH}
tar -xvf ${ZEPHYR_SDK_FILE_NAME} -C "${INSTALLATION_PATH}" --strip-components=1
rm ${ZEPHYR_SDK_FILE_NAME}

# Setup SDK
cd ${INSTALLATION_PATH}
./setup.sh -h
