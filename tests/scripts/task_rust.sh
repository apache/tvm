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

export TVM_HOME="$(git rev-parse --show-toplevel)"

export LD_LIBRARY_PATH="$TVM_HOME/lib:$TVM_HOME/build:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TVM_HOME/python":"$TVM_HOME/topi/python"
export RUST_DIR="$TVM_HOME/rust"
export LLVM_CONFIG_PATH=`which llvm-config-8`
echo "Using $LLVM_CONFIG_PATH"

cd $RUST_DIR
cargo fmt -- --check

# test common
cd $RUST_DIR/common
cargo build
cargo test --tests

cargo build --features bindings
cargo test --features bindings --tests

# test runtime
cd $RUST_DIR/runtime

# run basic tests
python3 tests/build_model.py
cargo test --tests

# run TVM module test
cd tests/test_tvm_basic
cargo run
cd -

cd tests/test_tvm_dso
cargo run
cd -

# # run wasm32 test
# cd tests/test_wasm32
# cargo build
# wasmtime $RUST_DIR/target/wasm32-wasi/debug/test-wasm32.wasm
# cd -

# run nn graph test
cd tests/test_nn
cargo run
cd -

# test frontend
cd $RUST_DIR/frontend

cargo test --tests -- --test-threads=1

# run basic tests on cpu
cd tests/basics
cargo build --features cpu
cargo run --features cpu
# uncomment when have more CI resources
# cargo build --features gpu
# cargo run --features gpu
# fi
cd -

# run callback tests separately: https://discuss.tvm.ai/t/are-global-functions-need-to-be-accessed-in-separate-processes/1075
cd tests/callback
cargo build
cargo run --bin int
cargo run --bin float
cargo run --bin array
cargo run --bin string
cd -

cd examples/resnet
cargo build
cd -
