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
cmake -DUSE_LLVM=ON -DINSTALL_DEV=ON -DCMAKE_INSTALL_PREFIX="$PREFIX" ..
make -j2 VERBOSE=1
make install
cd ..
