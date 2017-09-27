#!/bin/bash
mkdir -p docs/_build/html
# C++ doc
make doc

rm -rf python/nnvm/*.pyc python/nnvm/*/*.pyc

cd docs
PYTHONPATH=../python:../tvm/python:../tvm/topi/python make html || exit -1
cd _build/html
tar czf docs.tgz *
mv docs.tgz ../../../
