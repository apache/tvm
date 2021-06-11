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
#
#   Using this script we can reuse docker/install scripts to configure the reference 
#   virtual machine similar to CI QEMU setup.
#

set -e
set -x

source ~/.profile

# Init Zephyr
cd ~
# Using most recent commit that passes all the tests.
~/ubuntu_init_zephyr_project.sh ~/zephyr v2.5-branch --commit dabf23758417fd041fec2a2a821d8f526afac29d

# Build QEMU
sudo ~/ubuntu_install_qemu.sh --target-list arm-softmmu

# Cleanup
rm -f *.sh
