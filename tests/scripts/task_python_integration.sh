#!/bin/bash

set -e
set -u

export PYTHONPATH=python:topi/python:apps/extension/python
export LD_LIBRARY_PATH="build:${LD_LIBRARY_PATH:-}"

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc

# Test TVM
make cython
make cython3

# Test extern package
cd apps/extension
rm -rf lib
make
cd ../..
python -m nose -v apps/extension/tests

TVM_FFI=cython python -m nose -v tests/python/integration
TVM_FFI=ctypes python3 -m nose -v tests/python/integration
TVM_FFI=cython python -m nose -v tests/python/contrib
TVM_FFI=ctypes python3 -m nose -v tests/python/contrib

TVM_FFI=cython python -m nose -v tests/python/relay
TVM_FFI=ctypes python3 -m nose -v tests/python/relay

# Do not enable OpenGL
# TVM_FFI=cython python -m nose -v tests/webgl
# TVM_FFI=ctypes python3 -m nose -v tests/webgl
