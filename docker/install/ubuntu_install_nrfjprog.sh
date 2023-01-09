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
NRF_COMMANDLINE_TOOLS_SHA=5611536ca3377d64131ccd51232f9e33cde6d289b03ea33db0581a1288be8b0b10f995e2d60fdd4a3ce5a5c7b12bc85ddc672b282c9af8c5808707ab41543a7d

cd ~
mkdir -p ${TEMP_PATH_NAME}
wget --no-verbose -O $NRF_COMMANDLINE_TOOLS_FILE $NRF_COMMANDLINE_TOOLS_URL
echo "$NRF_COMMANDLINE_TOOLS_SHA $NRF_COMMANDLINE_TOOLS_FILE" | sha512sum --check

cd ${TEMP_PATH_NAME}
tar -xzvf "../${NRF_COMMANDLINE_TOOLS_FILE}"
apt-install-and-clear -y "./${JLINK_LINUX_INSTALLER}"

# Install nrfjprog
NRF_DEB_FILE=nrf-command-line-tools_amd64.deb
NRF_DEB_FILE_SHA=1f0339e16d50345ddde9757c2a4211361bcc78ff7371aac09decfffa809d86329001f5bc135f33dd154000a8f0da8bee4a0e80d3865ceff229f63ff9ace5ea95
wget --no-verbose -O $NRF_DEB_FILE https://nsscprodmedia.blob.core.windows.net/prod/software-and-other-downloads/desktop-software/nrf-command-line-tools/sw/versions-10-x-x/10-18-1/nrf-command-line-tools_10.18.1_amd64.deb
echo "$NRF_DEB_FILE_SHA $NRF_DEB_FILE" | sha512sum --check
apt-install-and-clear -y ./$NRF_DEB_FILE

cd ..
rm -rf ${TEMP_PATH_NAME} "${NRF_COMMANDLINE_TOOLS_FILE}"
