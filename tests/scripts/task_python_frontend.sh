#!/bin/bash

export PYTHONPATH=nnvm/python:python:topi/python
# to avoid openblas threading error
export OMP_NUM_THREADS=1

# Rebuild cython
make cython || exit -1
make cython3 || exit -1

echo "Running nnvm unittest..."
python -m nose -v nnvm/tests/python/unittest || exit -1
python3 -m nose -v nnvm/tests/python/unittest || exit -1

echo "Running nnvm compiler test..."
python3 -m nose -v nnvm/tests/python/compiler || exit -1

echo "Running nnvm ONNX frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/onnx || exit -1

echo "Running nnvm MXNet frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/mxnet || exit -1

echo "Running nnvm Keras frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/keras || exit -1

echo "Running nnvm Tensorflow frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/tensorflow || exit -1

echo "Running nnvm CoreML frontend test..."
python3 -m nose -v nnvm/tests/python/frontend/coreml || exit -1

echo "Running relay MXNet frontend test..."
python3 -m nose -v tests/python/frontend/mxnet || exit -1

echo "Running relay Keras frontend test..."
python3 -m nose -v tests/python/frontend/keras || exit -1

echo "Running relay ONNX frondend test..."
python3 -m nose -v tests/python/frontend/onnx || exit -1

echo "Running relay CoreML frondend test..."
python3 -m nose -v tests/python/frontend/coreml || exit -1

echo "Running relay Tensorflow frontend test..."
python3 -m nose -v tests/python/frontend/tensorflow || exit -1

echo "Running nnvm to relay frontend test..."
python3 -m nose -v tests/python/frontend/nnvm_to_relay || exit -1

echo "Running relay TFLite frontend test..."
python3 -m nose -v tests/python/frontend/tflite || exit -1

echo "Running relay caffe2 frondend test..."
python3 -m nose -v tests/python/frontend/caffe2 || exit -1
