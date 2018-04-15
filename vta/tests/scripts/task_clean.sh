#!/bin/bash
echo "Cleanup data..."
cd nnvm
make clean

cd tvm
make clean

cd ../..
make clean
