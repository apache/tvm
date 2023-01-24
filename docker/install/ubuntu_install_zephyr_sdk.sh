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

ZEPHYR_SDK_FILE_SHA=8e3572fbca9f9ba18a4436c00d680af34a85e239f7fe66c7988da85571a0d23d
ZEPHYR_SDK_FILE_NAME=zephyr-sdk-0.15.2_linux-x86_64.tar.gz
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.15.2/zephyr-sdk-0.15.2_linux-x86_64.tar.gz
echo "$ZEPHYR_SDK_FILE_SHA ${ZEPHYR_SDK_FILE_NAME}" | sha256sum --check

mkdir ${INSTALLATION_PATH}
tar -xvf ${ZEPHYR_SDK_FILE_NAME} -C "${INSTALLATION_PATH}" --strip-components=1
rm ${ZEPHYR_SDK_FILE_NAME}

# Setup SDK
cd ${INSTALLATION_PATH}
./setup.sh -h
