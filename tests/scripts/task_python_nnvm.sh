#!/bin/bash

export PYTHONPATH=nnvm/python:python:topi/python

echo "Running unittest..."
python -m nose -v nnvm/tests/python/unittest || exit -1
python3 -m nose -v nnvm/tests/python/unittest || exit -1

echo "Running compiler test..."
python -m nose -v nnvm/tests/python/compiler || exit -1
python3 -m nose -v nnvm/tests/python/compiler || exit -1

echo "Running ONNX frontend test..."
python -m nose -v nnvm/tests/python/frontend/onnx || exit -1

echo "Running MXNet frontend test..."
python -m nose -v nnvm/tests/python/frontend/mxnet || exit -1

echo "Running Keras frontend test..."
python -m nose -v nnvm/tests/python/frontend/keras || exit -1
