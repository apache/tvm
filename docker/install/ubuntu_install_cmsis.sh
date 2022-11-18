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
Usage: docker/install/ubuntu_install_cmsis.sh <INSTALLATION_PATH>
INSTALLATION_PATH is the installation path for the CMSIS.
EOF
}

if [ "$#" -lt 1 -o "$1" == "--help" -o "$1" == "-h" ]; then
    show_usage
    exit -1
fi

INSTALLATION_PATH=$1
shift

# Create installation path directory
mkdir -p "${INSTALLATION_PATH}"

# Download and extract CMSIS
CMSIS_SHA="51263182d16c92649a48144ba56c0945f9fce60e"
CMSIS_SHASUM="d02573e5a8908c741d8558f01be2939aae6e940933ccb58123fa972864947759eefe5d554688db3910c8ed665a248b477b5e4458e12773385c67f8a2136b3b34"
CMSIS_URL="http://github.com/ARM-software/CMSIS_5/archive/${CMSIS_SHA}.tar.gz"
DOWNLOAD_PATH="/tmp/${CMSIS_SHA}.tar.gz"

wget ${CMSIS_URL} -O "${DOWNLOAD_PATH}"
echo "$CMSIS_SHASUM" ${DOWNLOAD_PATH} | sha512sum -c
tar -xf "${DOWNLOAD_PATH}" -C "${INSTALLATION_PATH}" --strip-components=1
touch "${INSTALLATION_PATH}"/"${CMSIS_SHA}".sha
echo "SUCCESS"
