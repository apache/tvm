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

set -u
set -o pipefail

VULKAN_VERSION=1.4.309
UBUNTU_VERSION=jammy

wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc \
	| gpg --dearmor -o /usr/share/keyrings/lunarg-vulkan.gpg
wget -qO /tmp/lunarg-vulkan.list \
	http://packages.lunarg.com/vulkan/${VULKAN_VERSION}/lunarg-vulkan-${VULKAN_VERSION}-${UBUNTU_VERSION}.list
sed -E 's|^deb(-src)? |deb\1 [signed-by=/usr/share/keyrings/lunarg-vulkan.gpg] |' /tmp/lunarg-vulkan.list \
	> /etc/apt/sources.list.d/lunarg-vulkan-${VULKAN_VERSION}-${UBUNTU_VERSION}.list
rm -f /tmp/lunarg-vulkan.list
apt-get update
apt-install-and-clear -y vulkan-sdk
