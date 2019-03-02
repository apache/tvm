#!/bin/bash

set -e

if [ -z "$PREFIX" ]; then
  PREFIX="$CONDA_PREFIX"
fi

if [ -z "$cuda" ] || [ "$cuda" == "False" ]; then
    CUDA_OPT=""
else
    CUDA_OPT="-DUSE_CUDA=ON"
fi

rm -rf build || true
mkdir -p build
cd build
cmake $CUDA_OPT -DUSE_LLVM=ON -DINSTALL_DEV=ON -DCMAKE_INSTALL_PREFIX="$PREFIX" ..
make -j4 VERBOSE=1
make install
cd ..
