#!/bin/bash

set -e

export LD_LIBRARY_PATH=lib:$LD_LIBRARY_PATH

tvm_root="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$tvm_root/python":"$tvm_root/nnvm/python":"$tvm_root/topi/python"

cd rust
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
