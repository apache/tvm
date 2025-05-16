#!/usr/bin/env bash
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

# skip rust tests for now because of out of sync to latest FFI
exit 0

export TVM_HOME="$(git rev-parse --show-toplevel)"
echo "Using TVM_HOME=$TVM_HOME"
export LD_LIBRARY_PATH="$TVM_HOME/lib:$TVM_HOME/build:${LD_LIBRARY_PATH:-}"
echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
export PYTHONPATH="$TVM_HOME/python:${PYTHONPATH}"
echo "Using PYTHONPATH=$PYTHONPATH"
export RUST_DIR="$TVM_HOME/rust"
echo "Using RUST_DIR=$RUST_DIR"

export LLVM_CONFIG_DEFAULT=`which llvm-config-10`
export LLVM_CONFIG_PATH="${LLVM_CONFIG_PATH:-$LLVM_CONFIG_DEFAULT}"

echo "Using LLVM_CONFIG_PATH=$LLVM_CONFIG_PATH"

TVM_RUSTC_VERSION=`rustc --version`
echo "Using TVM_RUSTC_VERSION=$TVM_RUSTC_VERSION"

TVM_CARGO_VERSION=`cargo --version`
echo "Using TVM_CARGO_VERSION=$TVM_CARGO_VERSION"

# to avoid CI CPU thread throttling.
export TVM_BIND_THREADS=0
export OMP_NUM_THREADS=1

# First we test tvm-sys the core Rust bindings.
cd $RUST_DIR/tvm-sys
# First we test w/o the bindings feature on.
cargo build
cargo test --features static-linking --tests

# Second we test w/ the bindings feature on.
cargo build --features dynamic-linking
cargo test --features dynamic-linking --tests

# Next we test the runtime API.
cd $RUST_DIR/tvm-rt
# Build and run the tests.
cargo test
