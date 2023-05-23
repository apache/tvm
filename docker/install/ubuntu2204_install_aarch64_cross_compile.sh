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
# Used for debugging RVM build
set -x
set -o pipefail

architecture_type=$(uname -i)
if [ "$architecture_type" != "aarch64" ]; then
  # Install gcc and g++ for cross-compiling c++ on ubuntu
  apt-get update && apt-install-and-clear -y --no-install-recommends \
      g++-aarch64-linux-gnu \
      gcc-aarch64-linux-gnu \

  # Add Aarch64 packages to the apt sources list
  echo >> /etc/apt/sources.list.d/arm64.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ jammy main restricted" >> /etc/apt/sources.list.d/arm64.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ jammy-updates main restricted" >> /etc/apt/sources.list.d/arm64.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ jammy universe" >> /etc/apt/sources.list.d/arm64.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ jammy-updates universe" >> /etc/apt/sources.list.d/arm64.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ jammy multiverse" >> /etc/apt/sources.list.d/arm64.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ jammy-updates multiverse" >> /etc/apt/sources.list.d/arm64.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list.d/arm64.list

  # Fix apt-get update by specifying the amd64 architecture in sources.list
  sed -i -e 's/deb /deb [arch=amd64] /g' /etc/apt/sources.list

  # Install the required packages for cross-compiling
  dpkg --add-architecture arm64
  apt-get update && apt-install-and-clear -y --no-install-recommends \
      zlib1g-dev:arm64 \
      libtinfo-dev:arm64

fi
