#!/bin/bash
echo "Build TVM..."
make "$@"
cd nnvm

echo "Build NNVM..."
make "$@"
