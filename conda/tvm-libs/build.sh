#!/bin/bash

# Fix for OSX build to hide the clang LLVM
rm -f ${BUILD_PREFIX}/bin/llvm-config
rm -rf ${BUILD_PREFIX}/lib/cmake

set -e

if [ -z "$PREFIX" ]; then
  PREFIX="$CONDA_PREFIX"
fi

if [ -z "$cuda" ] || [ "$cuda" == "False" ]; then
    CUDA_OPT=""
else
    CUDA_OPT="-DUSE_CUDA=ON -DUSE_CUBLAS=ON"
fi

if [ "$target_platform" == "osx-64" ]; then
    # macOS 64 bits
    METAL_OPT=""  # Conda can only target 10.9 for now
    TOOLCHAIN_OPT=""
else
    METAL_OPT=""
    if [ "$target_platform" == "linux-64" ]; then
        # Linux 64 bits
        TOOLCHAIN_OPT="-DCMAKE_TOOLCHAIN_FILE=${RECIPE_DIR}/../cross-linux.cmake"
    else
        # Windows (or 32 bits, which we don't support)
        METAL_OPT=""
        TOOLCHAIN_OPT=""
    fi
fi

rm -rf build || true
mkdir -p build
cd build
cmake $METAL_OPT $CUDA_OPT -DUSE_LLVM=ON -DINSTALL_DEV=ON -DCMAKE_INSTALL_PREFIX="$PREFIX" $TOOLCHAIN_OPT ..
make -j${CPU_COUNT} VERBOSE=1
make install
cd ..
