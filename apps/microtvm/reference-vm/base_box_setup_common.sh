#!/bin/bash -e
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
set -x

# Fix network DNS issue
sudo sed -i 's/DNSSEC=yes/DNSSEC=no/' /etc/systemd/resolved.conf
sudo systemctl restart systemd-resolved

sudo apt update
sudo apt install -y build-essential
sudo apt-get --purge remove modemmanager  # required to access serial ports.

# Core
sudo ~/ubuntu_install_core.sh
rm -f ~/ubuntu_install_core.sh

sudo apt install -y --no-install-recommends git \
     gperf ccache dfu-util device-tree-compiler xz-utils file \
     gcc gcc-multilib g++-multilib libsdl2-dev

# Cmake
wget --no-verbose https://apt.kitware.com/keys/kitware-archive-latest.asc
sudo apt-key add kitware-archive-latest.asc
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt update
sudo apt install -y --no-install-recommends \
     cmake=3.22.2-0kitware1ubuntu18.04.1 cmake-data=3.22.2-0kitware1ubuntu18.04.1 \

# Python
sudo ~/ubuntu_install_python.sh
rm -f ~/ubuntu_install_python.sh

# Poetry deps
sudo apt install -y python3-venv

# TVM deps
# TODO(mehrdadh): replace with ubuntu_install_llvm.sh
sudo apt install -y llvm

# ONNX deps
sudo apt install -y protobuf-compiler libprotoc-dev

# Poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3

# Host name
OLD_HOSTNAME=$(hostname)
sudo hostnamectl set-hostname microtvm
sudo sed -i.bak "s/${OLD_HOSTNAME}/microtvm.localdomain/g" /etc/hosts
