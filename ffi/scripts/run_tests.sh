# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#!/bin/bash
set -euxo pipefail

BUILD_TYPE=RelWithDebugInfo

rm -rf build/CMakeFiles build/CMakeCache.txt
cmake -G Ninja -S . -B build  -DTVM_FFI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build build --parallel 16 --clean-first --config ${BUILD_TYPE} --target tvm_ffi_tests
GTEST_COLOR=1 ctest -V -C ${BUILD_TYPE} --test-dir build --output-on-failure
