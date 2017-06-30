#!/bin/bash
export PYTHONPATH=python:examples/extension/python
export PYTHONPATH=${PYTHONPATH}:examples/graph_executor/python:examples/graph_executor/nnvm/python
export LD_LIBRARY_PATH=lib:${LD_LIBRARY_PATH}

# Test TVM
make cython || exit -1

# Test extern package package
cd examples/extension
make || exit -1
cd ../..
python -m nose -v examples/extension/tests || exit -1

# Test NNVM integration
cd examples/graph_executor
make || exit -1
cd ../..
python -m nose -v examples/graph_executor/tests || exit -1

TVM_FFI=cython python -m nose -v tests/python/integration || exit -1
TVM_FFI=ctypes python3 -m nose -v tests/python/integration || exit -1
TVM_FFI=cython python -m nose -v tests/python/contrib || exit -1
TVM_FFI=ctypes python3 -m nose -v tests/python/contrib || exit -1
