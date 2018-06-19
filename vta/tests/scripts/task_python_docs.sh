#!/bin/bash
cd tvm
make cython
make cython3
cd ../

mkdir -p docs/_build/html
# C++ doc
make doc

rm -rf python/vta/*.pyc python/vta/*/*.pyc

cd docs
PYTHONPATH=../python:../tvm/python:../tvm/topi/python:../tvm/nnvm/python make html || exit -1
cd _build/html
tar czf docs.tgz *
mv docs.tgz ../../../
