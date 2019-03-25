#!/bin/bash

set -e
set -u

export PYTHONPATH=python:nnvm/python:vta/python:topi/python

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc python/tvm/*/*/*/*.pyc
rm -rf ~/.tvm

# Rebuild cython
make cython
make cython3

echo "Running unittest..."
python -m nose -v vta/tests/python/unittest
python3 -m nose -v vta/tests/python/unittest

echo "Running integration test..."
python -m nose -v vta/tests/python/integration
python3 -m nose -v vta/tests/python/integration
