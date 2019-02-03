#!/bin/bash

set -e

export TVM_HOME="$(git rev-parse --show-toplevel)"

export LD_LIBRARY_PATH="$TVM_HOME/lib":"$TVM_HOME/build":"$TVM_HOME/nnvm":$LD_LIBRARY_PATH
export PYTHONPATH="$TVM_HOME/python":"$TVM_HOME/nnvm/python":"$TVM_HOME/topi/python"
export RUST_DIR="$TVM_HOME/rust"

cd $RUST_DIR
cargo fmt -- --check

# test common
cd $RUST_DIR/common
cargo build --features runtime
cargo test --features runtime --tests

cargo build --features frontend
cargo test --features frontend --tests

# test runtime
cd $RUST_DIR/runtime

# run basic tests
python3 tests/build_model.py
cargo test --tests

# run TVM module test
cd tests/test_tvm_basic
cargo run
cd -

# run NNVM graph test
cd tests/test_nnvm
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
