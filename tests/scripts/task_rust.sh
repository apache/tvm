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

# Next we test the graph executor crate.
cd $RUST_DIR/tvm-graph-rt

# We first we compile a model using the Python bindings then run the tests.
python3 tests/build_model.py
cargo test --tests

# Run some more tests involving the graph executor API.
cd tests/test_tvm_basic
cargo run
cd -

cd tests/test_tvm_dso
cargo run
cd -

# run wasm32 test
# cd tests/test_wasm32
# cargo build
# wasmtime $RUST_DIR/target/wasm32-wasi/debug/test-wasm32.wasm
# cd -

# Disabled, see https://github.com/apache/tvm/issues/11419
# # run nn graph test
# cd tests/test_nn
# cargo run
# cd -

# Finally we test the TVM crate which provides both runtime
# and compiler bindings.
cd $RUST_DIR/tvm

cargo test

# run basic tests on cpu
cd tests/basics
cargo run --features cpu
# uncomment when have more CI resources
# cargo build --features gpu
# cargo run --features gpu
# fi
cd -

# TODO(@jroesch): I believe this is no longer true, refactor in follow up PR.
# run callback tests separately: https://discuss.tvm.ai/t/are-global-functions-need-to-be-accessed-in-separate-processes/1075
cd tests/callback
cargo build
cargo run --bin int
cargo run --bin float
cargo run --bin array
cargo run --bin string
cd -

cd examples/resnet
cargo run
cd -
