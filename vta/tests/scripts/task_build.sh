#!/bin/bash
echo "Build TVM..."
cd nnvm/tvm
cp make/config.mk .
echo USE_CUDA=0 >> config.mk
echo LLVM_CONFIG=llvm-config-4.0 >> config.mk
echo USE_RPC=1 >> config.mk
echo USE_BLAS=openblas >> config.mk
echo USE_GRAPH_RUNTIME=1 >> config.mk
make "$@"
make cython
make cython3
cd ..

echo "Build NNVM..."
make "$@"

cd ..

echo "Build VTA..."
make "$@"
