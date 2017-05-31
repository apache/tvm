#!/bin/bash
export PYTHONPATH=python:examples/extension
export LD_LIBRARY_PATH=lib:${LD_LIBRARY_PATH}

# Test extern package package
cd examples/extension
make || exit -1
cd ../..
python -m nose -v examples/extension/tests || exit -1

# Test TVM
make cython || exit -1
TVM_FFI=cython python -m nose -v tests/python/integration || exit -1
TVM_FFI=ctypes python3 -m nose -v tests/python/integration || exit -1
