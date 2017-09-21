#!/bin/bash
echo "Build the libraries.."
mkdir -p lib
make
echo "Run the example"
export LD_LIBRARY_PATH=../../lib
export DYLD_LIBRARY_PATH=../../lib
lib/cpp_deploy
