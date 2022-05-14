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

CMSIS_VER="5.8.0"

# Create installation path directory
mkdir -p "${INSTALLATION_PATH}"

# Download and extract CMSIS
cd "${HOME}"
wget --quiet "https://github.com/ARM-software/CMSIS_5/archive/${CMSIS_VER}.tar.gz"
tar -xf "${CMSIS_VER}.tar.gz" -C "${INSTALLATION_PATH}" --strip-components=1

# Remove tar file
rm -f "${CMSIS_VER}.tar.gz"
