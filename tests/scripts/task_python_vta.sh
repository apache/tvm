#!/bin/bash

export PYTHONPATH=python:nnvm/python:vta/python:topi/python

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc python/tvm/*/*/*/*.pyc
rm -rf ~/.tvm

# Rebuild cython
make cython || exit -1
make cython3 || exit -1

echo "Running unittest..."
python -m nose -v vta/tests/python/unittest || exit -1
python3 -m nose -v vta/tests/python/unittest || exit -1

echo "Running integration test..."
python -m nose -v vta/tests/python/integration || exit -1
python3 -m nose -v vta/tests/python/integration || exit -1
