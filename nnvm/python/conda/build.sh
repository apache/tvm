#!/bin/bash

set -e

if [ -z "$PREFIX" ]; then
  PREFIX="$CONDA_PREFIX"
fi

if [ "$(uname)" = 'Darwin' ]
then
    # Without this, Apple's default shipped clang will refuse to see any
    # headers like mutex.
    export MACOSX_DEPLOYMENT_TARGET=10.9
fi

rm -rf build || true
mkdir -p build
cd build
# Enable static-libstdc++ to make it easier to link this library with
# other C++ compilers
CXXFLAGS=-static-libstdc++ cmake -DCMAKE_PREFIX_PATH=${PREFIX} -DCMAKE_INSTALL_PREFIX=${PREFIX} ..
make -j4 VERBOSE=1
make install/fast
cd ..

cd python
$PYTHON setup.py install
cd ..
