#!/bin/bash

export PYTHONPATH=nnvm/python:python:topi/python
# to avoid openblas threading error
export OMP_NUM_THREADS=1

# Rebuild cython
make cython || exit -1
make cython3 || exit -1

echo "Running unittest..."
python -m nose -v nnvm/tests/python/unittest || exit -1
python3 -m nose -v nnvm/tests/python/unittest || exit -1

echo "Running compiler test..."
python -m nose -v nnvm/tests/python/compiler || exit -1
python3 -m nose -v nnvm/tests/python/compiler || exit -1

echo "Running ONNX frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/onnx || exit -1

echo "Running MXNet frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/mxnet || exit -1

echo "Running Keras frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/keras || exit -1

echo "Running Tensorflow frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/tensorflow || exit -1
