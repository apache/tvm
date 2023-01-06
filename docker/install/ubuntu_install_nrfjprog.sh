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
set -x

TEMP_PATH_NAME="nrfjprog_tmp_path"

# Install JLink
NRF_COMMANDLINE_TOOLS_FILE=nRFCommandLineToolsLinuxamd64.tar.gz
NRF_COMMANDLINE_TOOLS_URL=https://nsscprodmedia.blob.core.windows.net/prod/software-and-other-downloads/desktop-software/nrf-command-line-tools/sw/versions-10-x-x/10-18-1/nrf-command-line-tools-10.18.1_linux-amd64.tar.gz
JLINK_LINUX_INSTALLER=JLink_Linux_V780c_x86_64.deb

cd ~
mkdir -p ${TEMP_PATH_NAME}
wget --no-verbose -O $NRF_COMMANDLINE_TOOLS_FILE $NRF_COMMANDLINE_TOOLS_URL

cd ${TEMP_PATH_NAME}
tar -xzvf "../${NRF_COMMANDLINE_TOOLS_FILE}"
apt-install-and-clear -y "./${JLINK_LINUX_INSTALLER}"

# Install nrfjprog
wget --no-verbose https://nsscprodmedia.blob.core.windows.net/prod/software-and-other-downloads/desktop-software/nrf-command-line-tools/sw/versions-10-x-x/10-18-1/nrf-command-line-tools_10.18.1_amd64.deb
apt-install-and-clear -y ./nrf-command-line-tools_10.18.1_amd64.deb

cd ..
rm -rf ${TEMP_PATH_NAME} "${NRF_COMMANDLINE_TOOLS_FILE}"
