#!/bin/sh
mkdir -p build
cp cmake/config.cmake build/
cd build
#cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j

