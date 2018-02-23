#!/bin/bash
export PYTHONPATH=python:apps/extension/python
export LD_LIBRARY_PATH=lib:${LD_LIBRARY_PATH}

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc

# Test TVM
make cython || exit -1
make cython3 || exit -1

# Test extern package package
cd apps/extension
make || exit -1
cd ../..
python -m nose -v apps/extension/tests || exit -1

TVM_FFI=cython python -m nose -v tests/python/integration || exit -1
TVM_FFI=ctypes python3 -m nose -v tests/python/integration || exit -1
TVM_FFI=cython python -m nose -v tests/python/contrib || exit -1
TVM_FFI=ctypes python3 -m nose -v tests/python/contrib || exit -1

# Do not enabke OpenGL
# TVM_FFI=cython python -m nose -v tests/webgl || exit -1
# TVM_FFI=ctypes python3 -m nose -v tests/webgl || exit -1
