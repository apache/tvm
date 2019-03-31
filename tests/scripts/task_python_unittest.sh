#!/bin/bash

set -e
set -u

export PYTHONPATH=python:topi/python

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

TVM_FFI=ctypes python -m nose -v tests/python/unittest
TVM_FFI=ctypes python3 -m nose -v tests/python/unittest
make cython
make cython3
TVM_FFI=cython python -m nose -v tests/python/unittest
TVM_FFI=cython python3 -m nose -v tests/python/unittest
