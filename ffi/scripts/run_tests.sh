#!/bin/bash
set -euxo pipefail

BUILD_TYPE=RelWithDebugInfo

rm -rf build/CMakeFiles build/CMakeCache.txt
cmake -G Ninja -S . -B build  -DTVM_FFI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DTVM_FFI_BUILD_REGISTRY=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build --parallel 16 --clean-first --config ${BUILD_TYPE} --target tvm_ffi_tests
GTEST_COLOR=1 ctest -V -C ${BUILD_TYPE} --test-dir build --output-on-failure
