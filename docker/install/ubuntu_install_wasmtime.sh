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

set -euxo pipefail

# install wasmtime (note: requires ubuntu_install_rust.sh to run first)
apt-install-and-clear -y --no-install-recommends libc6-dev-i386
export WASMTIME_HOME=/opt/wasmtime
curl https://wasmtime.dev/install.sh -sSf | bash
export PATH="${WASMTIME_HOME}/bin:${PATH}"
rustup target add wasm32-wasip1

# make rust usable by all users after install during container build
chmod -R a+rw /opt/rust
