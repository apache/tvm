#!/bin/bash

export PYTHONPATH=python:topi/python:apps/extension/python
export LD_LIBRARY_PATH=build:${LD_LIBRARY_PATH}

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

make cython || exit -1
make cython3 || exit -1
TVM_FFI=cython python -m nose -v tests/python/relay/ir_relay_parser.py || exit -1
TVM_FFI=ctypes python3 -m nose -v tests/python/relay/ir_relay_parser.py || exit -1
