#!/usr/bin/env bash
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

set -euxo pipefail

BUILD_DIR=$1
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cp ../cmake/config.cmake .

echo set\(USE_SORT ON\) >> config.cmake
echo set\(USE_RELAY_DEBUG ON\) >> config.cmake
echo set\(CMAKE_BUILD_TYPE=Debug\) >> config.cmake
echo set\(CMAKE_CXX_FLAGS \"-Werror -Wp,-D_GLIBCXX_ASSERTIONS\"\) >> config.cmake
echo set\(HIDE_PRIVATE_SYMBOLS ON\) >> config.cmake
echo set\(USE_LIBBACKTRACE OFF\) >> config.cmake
echo set\(USE_CCACHE OFF\) >> config.cmake
echo set\(SUMMARIZE ON\) >> config.cmake

architecture_type=$(uname -i)
if [ "$architecture_type" != "aarch64" ]; then
  echo set\(USE_LLVM \"/usr/llvm-aarch64/bin/llvm-config --link-static\"\) >> config.cmake

  # Cross compile to aarch64
  echo set\(CMAKE_C_COMPILER aarch64-linux-gnu-gcc\) >> config.cmake
  echo set\(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++\) >> config.cmake

  echo set\(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu\) >> config.cmake
  echo set\(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER\) >> config.cmake
  echo set\(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY\) >> config.cmake
  echo set\(ZLIB_LIBRARY /lib/aarch64-linux-gnu/libz.a\) >> config.cmake
else
  # This usually runs in the ci_arm docker image.
  echo -e 'find_program(LLVM_CONFIG "llvm-config")\nif (LLVM_CONFIG) \n\tset(USE_LLVM llvm-config) \nelse() \n\tset(USE_LLVM llvm-config-15)\nendif()' >> config.cmake
fi
