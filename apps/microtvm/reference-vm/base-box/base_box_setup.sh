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

skip_zeroing_disk=0
if [ -e "$HOME/skip_zeroing_disk" ]; then
    echo "NOTE: will not zero disk at the end due to VMWare Fusion bug"
    echo "See: https://communities.vmware.com/t5/VMware-Fusion-Discussions/VMWare-Fusion-Pro-11-15-6-16696540-causes-macOS-crash-during/m-p/2284011#M139190"
    skip_zeroing_disk=1
fi

# Install common configs
~/base_box_setup_common.sh
rm -f ~/base_box_setup_common.sh

# Poetry
sed -i "/^# If not running interactively,/ i source \$HOME/.poetry/env" ~/.bashrc
sed -i "/^# If not running interactively,/ i export ZEPHYR_BASE=$HOME/zephyr/zephyr" ~/.bashrc
sed -i "/^# If not running interactively,/ i\\ " ~/.bashrc

# nrfjprog
sudo ~/ubuntu_install_nrfjprog.sh
rm -f ~/ubuntu_install_nrfjprog.sh

# Zephyr
pip3 install --user -U west
echo 'export PATH=$HOME/.local/bin:"$PATH"' >> ~/.profile
source ~/.profile
echo PATH=$PATH

cd ~
echo "Downloading zephyr SDK..."
~/ubuntu_install_zephyr_sdk.sh ~/zephyr-sdk
rm -f ubuntu_install_zephyr_sdk.sh

# GDB for Zephyr SDK depends on python3.8
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.8-dev

sudo find ~/zephyr-sdk -name '*.rules' -exec cp {} /etc/udev/rules.d \;
sudo udevadm control --reload

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
