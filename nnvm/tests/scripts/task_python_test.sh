#!/bin/bash

export PYTHONPATH=python:tvm/python:tvm/topi/python

echo "Running unittest..."
python -m nose -v tests/python/unittest || exit -1
python3 -m nose -v tests/python/unittest || exit -1

echo "Running compiler test..."
python -m nose -v tests/python/compiler || exit -1
python3 -m nose -v tests/python/compiler || exit -1
