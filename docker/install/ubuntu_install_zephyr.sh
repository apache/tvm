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

pip3 install west

# Init ZephyrProject
# To keep in sync with the version 
# defined in apps/microtvm/zephyr/template_project/microtvm_api_server.py
# We use `-branch` tag since it tracks the same version with extra patches for bugs.
ZEPHYR_VERSION="v2.5-branch"
ZEPHYR_PROJECT_PATH=/opt/zephyrproject
ZEPHYR_INIT_SCRIPT=$(find -name "ubuntu_init_zephyr_project.sh")
bash ${ZEPHYR_INIT_SCRIPT} ${ZEPHYR_PROJECT_PATH} ${ZEPHYR_VERSION}
cd ${ZEPHYR_PROJECT_PATH}

# As part of the build process, Zephyr needs to touch some symlinks in zephyr/misc/generated/syscalls_links (this path is relative to the
# build directory for a project). Mark the zephyr installation world-writable since this is a docker
# container
chmod -R o+w ${ZEPHYR_PROJECT_PATH}

# This step is required because of the way docker/bash.sh works. It sets the user home directory to
# /workspace (or the TVM root, anyhow), and this means that zephyr expects a ~/.cache directory to be
# present *in the TVM project root*. Since we don't intend to add one to avoid dirtying the repo
# tree, we need to populate the zephyr fallback cache directory and ensure it's writable. Cache
# artifacts aren't intended to be saved into the docker image.
mkdir zephyr/.cache
chmod o+rwx zephyr/.cache

#/opt/west/bin/pip3 install -r /opt/zephyrproject/zephyr/scripts/requirements.txt
pip3 install -r /opt/zephyrproject/zephyr/scripts/requirements.txt

ZEPHYR_SDK_VERSION=0.12.3
ZEPHYR_SDK_FILE=zephyr-sdk-linux-setup.run
wget --no-verbose -O $ZEPHYR_SDK_FILE \
    https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v${ZEPHYR_SDK_VERSION}/zephyr-sdk-${ZEPHYR_SDK_VERSION}-x86_64-linux-setup.run
chmod +x $ZEPHYR_SDK_FILE
"./$ZEPHYR_SDK_FILE" -- -d /opt/zephyr-sdk
