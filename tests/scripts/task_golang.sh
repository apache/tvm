#!/bin/bash

set -e

export LD_LIBRARY_PATH=lib:$LD_LIBRARY_PATH

tvm_root="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$tvm_root/python":"$tvm_root/nnvm/python":"$tvm_root/topi/python"

# Golang tests
make -C golang tests
