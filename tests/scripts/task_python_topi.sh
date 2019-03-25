#!/bin/bash

set -e
set -u

export PYTHONPATH=python:topi/python

# Rebuild cython
make cython
make cython3

rm -rf python/tvm/*.pyc python/tvm/*/*.pyc python/tvm/*/*/*.pyc
rm -rf topi/python/topi/*.pyc topi/python/topi/*/*.pyc topi/python/topi/*/*/*.pyc topi/python/topi/*/*/*/*.pyc 

python -m nose -v topi/tests/python
python3 -m nose -v topi/tests/python
