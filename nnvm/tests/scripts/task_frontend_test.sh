#!/bin/bash

export PYTHONPATH=python:tvm/python:tvm/topi/python

echo "Running ONNX frontend test..."
python -m nose -v tests/python/frontend/onnx || exit -1

echo "Running MXNet frontend test..."
python -m nose -v tests/python/frontend/mxnet || exit -1

echo "Running Keras frontend test..."
python -m nose -v tests/python/frontend/keras || exit -1
