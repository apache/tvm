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

# Poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
sed -i "/^# If not running interactively,/ i source \$HOME/.poetry/env" ~/.bashrc
sed -i "/^# If not running interactively,/ i export ZEPHYR_BASE=$HOME/zephyr/zephyr" ~/.bashrc
sed -i "/^# If not running interactively,/ i\\ " ~/.bashrc
