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

export PYXIR_HOME=/opt/pyxir
mkdir "$PYXIR_HOME"

# install libraries for building Vitis-AI on ubuntu
apt-get update && apt-install-and-clear -y \
    graphviz \
    gnupg2 \
    gpg-agent \
    gcc-aarch64-linux-gnu


. $VAI_ROOT/conda/etc/profile.d/conda.sh
conda activate vitis-ai-tensorflow
pip3 install progressbar h5py==2.10.0

git clone --recursive --branch rel-v0.3.1 --depth 1 https://github.com/Xilinx/pyxir.git "${PYXIR_HOME}"
cd "${PYXIR_HOME}" && python3 setup.py install --use_vart_cloud_dpu --use_dpuczdx8g_vart
