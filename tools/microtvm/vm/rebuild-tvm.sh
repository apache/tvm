#!/bin/bash -e

set -e

cd "$(dirname $0)"
cd "$(git rev-parse --show-toplevel)"
BUILD_DIR=build-microtvm

if [ ! -e "${BUILD_DIR}" ]; then
    mkdir "${BUILD_DIR}"
fi
cp cmake/config.cmake "${BUILD_DIR}"
cd "${BUILD_DIR}"
sed -i 's/USE_MICRO OFF/USE_MICRO ON/' config.cmake
sed -i 's/USE_LLVM OFF/USE_LLVM ON/' config.cmake
cmake ..
make -j4
