#!/bin/bash
export LD_LIBRARY_PATH=lib:${LD_LIBRARY_PATH}
export PYTHONPATH=python:nnvm/python:topi/python

set -e

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
