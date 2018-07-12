#!/bin/bash

export PYTHONPATH=python:nnvm/python:vta/python:topi/python

echo "Running unittest..."
python -m nose -v vta/tests/python/unittest || exit -1
python3 -m nose -v vta/tests/python/unittest || exit -1

echo "Running integration test..."
python -m nose -v vta/tests/python/integration || exit -1
python3 -m nose -v vta/tests/python/integration || exit -1
