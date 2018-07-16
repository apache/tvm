#!/bin/bash

set -e

if [ -z "$PREFIX" ]; then
  PREFIX="$CONDA_PREFIX"
fi

rm -rf build || true
mkdir -p build
cd build
cmake -DUSE_LLVM=ON -DINSTALL_DEV=ON -DCMAKE_INSTALL_PREFIX="$PREFIX" ..
make -j2 VERBOSE=1
make install
cd ..
