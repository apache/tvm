#!/bin/bash

set -e

# See Note [CUDA_TOOLKIT_ROOT_DIR versus CUDA_BIN_PATH]
if [ -z "$CONDA_CUDA_HOME" ]; then
  CUDA_ARGS=""
else
  # See Note [Bash argument quoting]
  CUDA_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=$(printf %q "$CONDA_CUDA_HOME")"
fi

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
CXXFLAGS=-static-libstdc++ cmake -DCMAKE_PREFIX_PATH=${PREFIX} -DCMAKE_INSTALL_PREFIX=${PREFIX} -DUSE_CUDA=1 -DUSE_LLVM=1 -DINSTALL_DEV=1 $CUDA_ARGS ..
make -j20 VERBOSE=1
make install/fast
cd ..

# Also install the headers for libraries that TVM vendored
mkdir -p "$PREFIX/include"
# TODO: arguably dlpack and dmlc-core should get its own packaging and
# install their headers themselves
cp -R dlpack/include/. "$PREFIX/include"
cp -R dmlc-core/include/. "$PREFIX/include"
# TODO: HalideIR's includes could conflict, but TVM currently assumes they
# are installed here, awfully enough
cp -R HalideIR/src/. "$PREFIX/include"

cd python
$PYTHON setup.py install
cd ..
