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

#mkdir /home/tvm/.ssh
#cp ~vagrant/.ssh/authorized_keys /home/tvm/.ssh
#chown tvm:tvm /home/tvm/.ssh
#chown tvm:tvm /home/tvm/.ssh/authorized_keys
#chmod 644 /home/tvm/.ssh/authorized_keys
#chmod 755 /home/tvm/.ssh

#cp ~vagrant/setup-workspace.sh /home/tvm/setup-workspace.sh
#chown tvm:tvm /home/tvm/setup-workspace.sh
#chmod u+x /home/tvm/setup-workspace.sh
#sudo -u tvm -sH bash --login ~tvm/setup-workspace.sh "${TVM_HOME}"

set -e

# TVM
# NOTE: TVM is presumed to be mounted already by Vagrantfile.
cd "${TVM_HOME}"

apps/microtvm/reference-vm/zephyr/rebuild-tvm.sh

# NOTE: until the dependencies make it into a top-level pyproject.toml file in main,
# use this approach.
cp apps/microtvm/reference-vm/zephyr/pyproject.toml .

poetry lock
poetry install
poetry run pip3 install -r ~/zephyr/zephyr/scripts/requirements.txt

echo "export TVM_LIBRARY_PATH=\"$TVM_HOME\"/build-microtvm" >>~/.profile
