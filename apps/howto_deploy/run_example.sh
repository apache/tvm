#!/bin/bash
echo "Build the libraries.."
mkdir -p lib
make
echo "Run the example"
export LD_LIBRARY_PATH=../../lib:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=../../lib:${DYLD_LIBRARY_PATH}

echo "Run the deployment with all in one packed library..."
lib/cpp_deploy_pack

echo "Run the deployment with all in normal library..."
lib/cpp_deploy_normal
