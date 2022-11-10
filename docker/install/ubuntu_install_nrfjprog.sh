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

NRF_COMMANDLINE_TOOLS_FILE=nRFCommandLineToolsLinuxamd64.tar.gz
NRF_COMMANDLINE_TOOLS_URL=https://www.nordicsemi.com/-/media/Software-and-other-downloads/Desktop-software/nRF-command-line-tools/sw/Versions-10-x-x/10-12-1/nRFCommandLineTools10121Linuxamd64.tar.gz
NRF_COMMANDLINE_TOOLS_INSTALLER=nRF-Command-Line-Tools_10_12_1_Linux-amd64.deb
JLINK_LINUX_INSTALLER=JLink_Linux_V688a_x86_64.deb

cd ~
mkdir -p nrfjprog
wget --no-verbose -O $NRF_COMMANDLINE_TOOLS_FILE $NRF_COMMANDLINE_TOOLS_URL

cd nrfjprog
tar -xzvf "../${NRF_COMMANDLINE_TOOLS_FILE}"
apt-install-and-clear -y "./${JLINK_LINUX_INSTALLER}"
apt-install-and-clear -y "./${NRF_COMMANDLINE_TOOLS_INSTALLER}"

cd ..
rm -rf nrfjprog "${NRF_COMMANDLINE_TOOLS_FILE}"
