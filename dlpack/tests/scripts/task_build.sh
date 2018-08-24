#!/bin/bash
make || exit -1
mkdir -p build
cd build
cmake .. || exit -1
make || exit -1
