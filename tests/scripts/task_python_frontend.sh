#!/bin/bash

set -e
set -u

export PYTHONPATH=nnvm/python:python:topi/python
# to avoid openblas threading error
export OMP_NUM_THREADS=1

# Rebuild cython
make cython
make cython3

echo "Running relay TFLite frontend test..."
python3 -m nose -v tests/python/frontend/tflite

echo "Running nnvm unittest..."
python -m nose -v nnvm/tests/python/unittest
python3 -m nose -v nnvm/tests/python/unittest

echo "Running nnvm compiler test..."
python3 -m nose -v nnvm/tests/python/compiler

echo "Running nnvm ONNX frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/onnx

echo "Running nnvm MXNet frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/mxnet

echo "Running nnvm Keras frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/keras

echo "Running nnvm Tensorflow frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/tensorflow

echo "Running nnvm CoreML frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/coreml

echo "Running relay MXNet frontend test..."
python3 -m nose -v tests/python/frontend/mxnet

echo "Running relay Keras frontend test..."
python3 -m nose -v tests/python/frontend/keras

echo "Running relay ONNX frondend test..."
python3 -m nose -v tests/python/frontend/onnx

echo "Running relay CoreML frondend test..."
python3 -m nose -v tests/python/frontend/coreml

echo "Running nnvm to relay frontend test..."
python3 -m nose -v tests/python/frontend/nnvm_to_relay

echo "Running relay Tensorflow frontend test..."
python3 -m nose -v tests/python/frontend/tensorflow

echo "Running relay caffe2 frondend test..."
python3 -m nose -v tests/python/frontend/caffe2
