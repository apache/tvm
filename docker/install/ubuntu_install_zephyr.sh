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

cat <<EOF | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
-----BEGIN PGP PUBLIC KEY BLOCK-----

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
=0GnF
-----END PGP PUBLIC KEY BLOCK-----

EOF

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
