#!/bin/bash

set -e

if [ "$(uname)" = 'Darwin' ]
then
    # Without this, Apple's default shipped clang will refuse to see any
    # headers like mutex.
    export MACOSX_DEPLOYMENT_TARGET=10.9
fi

rm -rf build || true
mkdir -p build
cd build
cmake ..
make -j4 VERBOSE=1 nnvm-libs
cd ..

cd nnvm/python
$PYTHON setup.py install
cd ..
