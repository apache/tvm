#!/bin/bash
echo "Build TVM..."
cd tvm
cp cmake/config.cmake .
echo set\(USE_LLVM llvm-config-5.0\) >> config.cmake
echo set\(USE_RPC ON\) >> config.cmake
echo set\(USE_BLAS openblas\) >> config.cmake
echo set\(USE_GRAPH_RUNTIME ON\) >> config.cmake
make "$@"
make cython
make cython3
cd ..

echo "Build VTA..."
make "$@"
