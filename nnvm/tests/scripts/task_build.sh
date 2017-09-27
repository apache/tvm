#!/bin/bash
echo "Build TVM..."
cd tvm
cp make/config.mk .
echo USE_CUDNN=1 >> config.mk
echo USE_CUDA=1 >> config.mk
echo USE_OPENCL=1 >> config.mk
echo LLVM_CONFIG=llvm-config-4.0 >> config.mk
echo USE_RPC=1 >> config.mk
echo USE_BLAS=openblas >> config.mk
echo USE_GRAPH_RUNTIME=1 >> config.mk
make "$@"
cd ..

echo "Build NNVM..."
make "$@"
