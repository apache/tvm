#!/bin/bash

set -e

export LD_LIBRARY_PATH=lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH

export TVM_HOME="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$TVM_HOME/python":"$TVM_HOME/nnvm/python":"$TVM_HOME/topi/python"
export RUST_DIR="$TVM_HOME/rust"

# test common
cd $RUST_DIR/common
cargo fmt -- --check

cargo build --features runtime
cargo test --features runtime --tests

cargo build --features frontend
cargo test --features frontend --tests
cd -

# test runtime
cd $RUST_DIR/runtime
cargo fmt -- --check

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
cargo fmt -- --check

# run unit tests
cargo build
cargo test
cd -

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
cargo run --bin error
cd -

# run resnet example
cd examples/resnet
cargo build
cargo run
cd -
