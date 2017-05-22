#!/bin/bash

export PYTHONPATH=python

make cython || exit -1
TVM_FFI=cython python -m nose -v tests/python/integration || exit -1
TVM_FFI=ctypes python3 -m nose -v tests/python/integration || exit -1
