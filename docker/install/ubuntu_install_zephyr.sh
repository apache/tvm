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

sudo apt-install-and-clear -y --no-install-recommends \
     libsdl2-dev ca-certificates gnupg software-properties-common wget \
     git ninja-build gperf \
     ccache dfu-util device-tree-compiler wget \
     python3-dev python3-pip python3-setuptools python3-tk python3-wheel python3-venv \
     xz-utils file make gcc gcc-multilib g++-multilib apt-transport-https libudev-dev \
     libmagic1

wget --no-verbose https://apt.kitware.com/keys/kitware-archive-latest.asc
sudo apt-key add kitware-archive-latest.asc

echo deb https://apt.kitware.com/ubuntu/ jammy main\
     >> /etc/apt/sources.list.d/kitware.list
sudo apt-get update

sudo apt-install-and-clear -y cmake

# Find release version
apt-get update
apt-install-and-clear -y \
    lsb-core

release=$(lsb_release -sc)
if [ "${release}" == "bionic" ]; then
     python_cmd="python3"
elif [ "${release}" == "focal" ]; then
     python_cmd="python3.9"
elif [ "${release}" == "jammy" ]; then
     python_cmd="python3.9"
else
    echo "Don't know which version of python to use for Zephyr."
    exit 2
fi

# Current Zephyr version is compatible with python3.8.
# We use a different python env for Zephyr to test the
# real world scenario where TVM and Zephyr could be in different
# python environments.
# TODO: use virtual env for Zephyr.
$python_cmd -m pip install west

# Init ZephyrProject
ZEPHYR_PROJECT_PATH=/opt/zephyrproject
bash /install/ubuntu_init_zephyr_project.sh ${ZEPHYR_PROJECT_PATH} --branch v3.6-branch
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

$python_cmd -m pip install -r /opt/zephyrproject/zephyr/scripts/requirements.txt
