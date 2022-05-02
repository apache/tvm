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

export RUSTUP_HOME=/opt/rust
export CARGO_HOME=/opt/rust

# this rustc is one supported by the installed version of rust-sgx-sdk
HOST_ARG=
if [ "$(getconf LONG_BIT)" == "32" ]; then
    # When building in the i386 docker image on a 64-bit host, rustup doesn't
    # correctly detect the arch to install for so set it manually
    HOST_ARG="--default-host i686-unknown-linux-gnu"
fi

# shellcheck disable=SC2086 # word splitting is intentional here
curl -s -S -L https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path --profile minimal --default-toolchain stable $HOST_ARG
export PATH=$CARGO_HOME/bin:$PATH
rustup component add rustfmt
rustup component add clippy

# make rust usable by all users after install during container build
chmod -R a+rw /opt/rust
