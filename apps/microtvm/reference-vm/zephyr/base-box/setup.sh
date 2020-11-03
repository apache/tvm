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

sudo apt update
sudo apt install -y build-essential
sudo apt-get --purge remove modemmanager  # required to access serial ports.

# Zephyr
wget --no-verbose https://apt.kitware.com/keys/kitware-archive-latest.asc
sudo apt-key add kitware-archive-latest.asc
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt update
sudo apt install -y --no-install-recommends git cmake ninja-build gperf \
  ccache dfu-util device-tree-compiler wget \
  python3-dev python3-pip python3-setuptools python3-tk python3-wheel xz-utils file \
  make gcc gcc-multilib g++-multilib libsdl2-dev

# Avahi, so that ssh microtvm works.
# apt install -y avahi-daemon

OLD_HOSTNAME=$(hostname)
sudo hostnamectl set-hostname microtvm
sudo sed -i.bak "s/${OLD_HOSTNAME}/microtvm.localdomain/g" /etc/hosts

# Poetry deps
sudo apt install -y python3-venv

# TVM deps
sudo apt install -y llvm

# ONNX deps
sudo apt install -y protobuf-compiler libprotoc-dev

# nrfjprog
cd ~
mkdir -p nrfjprog
wget --no-verbose -O nRFCommandLineTools1090Linuxamd64.tar.gz https://www.nordicsemi.com/-/media/Software-and-other-downloads/Desktop-software/nRF-command-line-tools/sw/Versions-10-x-x/10-9-0/nRFCommandLineTools1090Linuxamd64tar.gz
cd nrfjprog
tar -xzvf ../nRFCommandLineTools1090Linuxamd64.tar.gz
sudo apt install -y ./JLink_Linux_V680a_x86_64.deb
sudo apt install -y ./nRF-Command-Line-Tools_10_9_0_Linux-amd64.deb
source ~/.profile
nrfjprog --help
cd ..
rm -rf nrfjprog nRFCommandLineTools1090Linuxamd64.tar.gz

# Zephyr
pip3 install --user -U west
echo 'export PATH=$HOME/.local/bin:"$PATH"' >> ~/.profile
source ~/.profile
echo PATH=$PATH
west init --mr v2.4.0 ~/zephyr
cd ~/zephyr
west update
west zephyr-export

cd ~
echo "Downloading zephyr SDK..."
wget --no-verbose https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.11.3/zephyr-sdk-0.11.3-setup.run
chmod +x zephyr-sdk-0.11.3-setup.run
./zephyr-sdk-0.11.3-setup.run -- -d ~/zephyr-sdk -y
rm -rf zephyr-sdk-0.11.3-setup.run

# GDB for Zephyr SDK depends on python3.8
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.8-dev

sudo find ~/zephyr-sdk -name '*.rules' -exec cp {} /etc/udev/rules.d \;
sudo udevadm control --reload

# Poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
sed -i "/^# If not running interactively,/ i source \$HOME/.poetry/env" ~/.bashrc
sed -i "/^# If not running interactively,/ i export ZEPHYR_BASE=$HOME/zephyr/zephyr" ~/.bashrc
sed -i "/^# If not running interactively,/ i\\ " ~/.bashrc

# Clean box for packaging as a base box
sudo apt-get clean
EMPTY_FILE="$HOME/EMPTY"
dd if=/dev/zero "of=${EMPTY_FILE}" bs=1M || /bin/true
if [ ! -e "${EMPTY_FILE}" ]; then
    echo "failed to zero empty sectors on disk"
    exit 2
fi
rm -f "${EMPTY_FILE}"
