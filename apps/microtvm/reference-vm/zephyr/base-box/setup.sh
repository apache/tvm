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

skip_zeroing_disk=0
if [ -e "$HOME/skip_zeroing_disk" ]; then
    echo "NOTE: will not zero disk at the end due to VMWare Fusion bug"
    echo "See: https://communities.vmware.com/t5/VMware-Fusion-Discussions/VMWare-Fusion-Pro-11-15-6-16696540-causes-macOS-crash-during/m-p/2284011#M139190"
    skip_zeroing_disk=1
fi

sudo apt update
sudo apt install -y build-essential
sudo apt-get --purge remove modemmanager  # required to access serial ports.

# Zephyr
wget --no-verbose https://apt.kitware.com/keys/kitware-archive-latest.asc
sudo apt-key add kitware-archive-latest.asc
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt update
# NOTE: latest cmake cannot be installed due to
# https://github.com/zephyrproject-rtos/zephyr/issues/30232
sudo apt install -y --no-install-recommends git \
     cmake=3.18.4-0kitware1 cmake-data=3.18.4-0kitware1 \
     ninja-build gperf ccache dfu-util device-tree-compiler wget \
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
NRF_COMMANDLINE_TOOLS_FILE=nRFCommandLineToolsLinuxamd64.tar.gz
NRF_COMMANDLINE_TOOLS_URL=https://www.nordicsemi.com/-/media/Software-and-other-downloads/Desktop-software/nRF-command-line-tools/sw/Versions-10-x-x/10-12-1/nRFCommandLineTools10121Linuxamd64.tar.gz
NRF_COMMANDLINE_TOOLS_INSTALLER=nRF-Command-Line-Tools_10_12_1_Linux-amd64.deb
JLINK_LINUX_INSTALLER=JLink_Linux_V688a_x86_64.deb

cd ~
mkdir -p nrfjprog
wget --no-verbose -O $NRF_COMMANDLINE_TOOLS_FILE $NRF_COMMANDLINE_TOOLS_URL
cd nrfjprog
tar -xzvf "../${NRF_COMMANDLINE_TOOLS_FILE}"
sudo apt install -y "./${JLINK_LINUX_INSTALLER}"
sudo apt install -y "./${NRF_COMMANDLINE_TOOLS_INSTALLER}"
source ~/.profile
nrfjprog --help
cd ..
rm -rf nrfjprog "${NRF_COMMANDLINE_TOOLS_FILE}"

# Zephyr
pip3 install --user -U west
echo 'export PATH=$HOME/.local/bin:"$PATH"' >> ~/.profile
source ~/.profile
echo PATH=$PATH
west init --mr v2.5.0 ~/zephyr
cd ~/zephyr
west update
west zephyr-export

cd ~
echo "Downloading zephyr SDK..."
ZEPHYR_SDK_VERSION=0.12.3
ZEPHYR_SDK_FILE=zephyr-sdk-linux-setup.run
wget --no-verbose -O $ZEPHYR_SDK_FILE \
    https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v${ZEPHYR_SDK_VERSION}/zephyr-sdk-${ZEPHYR_SDK_VERSION}-x86_64-linux-setup.run
chmod +x $ZEPHYR_SDK_FILE
"./$ZEPHYR_SDK_FILE" -- -d ~/zephyr-sdk -y
rm -rf ZEPHYR_SDK_FILE

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
if [ $skip_zeroing_disk -eq 0 ]; then
    echo "Zeroing disk..."
    EMPTY_FILE="$HOME/EMPTY"
    dd if=/dev/zero "of=${EMPTY_FILE}" bs=1M || /bin/true
    if [ ! -e "${EMPTY_FILE}" ]; then
        echo "failed to zero empty sectors on disk"
        exit 2
    fi
    rm -f "${EMPTY_FILE}"
else
    echo "NOTE: skipping zeroing disk due to command-line argument."
fi
