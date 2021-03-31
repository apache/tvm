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

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC
sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
echo $TZ > /etc/timezone

sudo apt-get install -y --no-install-recommends \
     libsdl2-dev ca-certificates gnupg software-properties-common wget \
     git cmake ninja-build gperf \
     ccache dfu-util device-tree-compiler wget \
     python3-dev python3-pip python3-setuptools python3-tk python3-wheel python3-venv \
     xz-utils file make gcc gcc-multilib g++-multilib apt-transport-https

wget --no-verbose https://apt.kitware.com/keys/kitware-archive-latest.asc
sudo apt-key add kitware-archive-latest.asc

sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update

sudo apt-get install -y cmake

#mkdir /opt/west
#python3.6 -mvenv /opt/west  # NOTE: include .6 to make a python3.6 link for west/cmake.
#/opt/west/bin/pip3 install west
pip3 install west

#cat <<EOF | tee /usr/local/bin/west >/dev/null
##!/bin/bash -e
#
#source /opt/west/bin/activate
#export ZEPHYR_BASE=/opt/zephyrproject/zephyr
#west "\$@"
#EOF
#chmod a+x /usr/local/bin/west

west init --mr v2.4.0 /opt/zephyrproject
cd /opt/zephyrproject
west update

# This step is required because of the way docker/bash.sh works. It sets the user home directory to
# /workspace (or the TVM root, anyhow), and this means that zephyr expects a ~/.cache directory to be
# present *in the TVM project root*. Since we don't intend to add one to avoid dirtying the repo
# tree, we need to populate the zephyr fallback cache directory and ensure it's writable. Cache
# artifacts aren't intended to be saved into the docker image.
mkdir zephyr/.cache
chmod o+rwx zephyr/.cache

west zephyr-export

#/opt/west/bin/pip3 install -r /opt/zephyrproject/zephyr/scripts/requirements.txt
pip3 install -r /opt/zephyrproject/zephyr/scripts/requirements.txt

SDK_VERSION=0.11.3
wget --no-verbose \
     https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v${SDK_VERSION}/zephyr-sdk-${SDK_VERSION}-setup.run
chmod +x zephyr-sdk-${SDK_VERSION}-setup.run
./zephyr-sdk-${SDK_VERSION}-setup.run -- -d /opt/zephyr-sdk
