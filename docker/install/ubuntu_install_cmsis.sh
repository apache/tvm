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

CMSIS_TAG="5.9.0"
CMSIS_NN_TAG="v4.1.0"

CMSIS_URL="https://github.com/ARM-software/CMSIS_5.git"
git clone ${CMSIS_URL} --branch ${CMSIS_TAG} --single-branch ${INSTALLATION_PATH}

CMSIS_NN_URL="https://github.com/ARM-software/CMSIS-NN.git"
git clone ${CMSIS_NN_URL} --branch ${CMSIS_NN_TAG} --single-branch ${INSTALLATION_PATH}/CMSIS-NN

touch "${INSTALLATION_PATH}"/"CMSIS_${CMSIS_TAG}".sha
touch "${INSTALLATION_PATH}"/"CMSIS_NN_${CMSIS_NN_TAG}".sha
echo "SUCCESS"
