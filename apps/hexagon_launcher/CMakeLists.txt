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

cmake_minimum_required(VERSION 3.2)
project(HexagonLauncher C CXX)

include(ExternalProject)

set(LAUNCHER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(VARS_NEEDED
  ANDROID_ABI
  ANDROID_PLATFORM
  USE_ANDROID_TOOLCHAIN
  USE_HEXAGON_ARCH
  USE_HEXAGON_SDK
  USE_HEXAGON_TOOLCHAIN
)
foreach(V IN LISTS VARS_NEEDED)
  if(NOT ${V})
    message(SEND_ERROR "Please set ${V}")
  endif()
endforeach()


ExternalProject_Add(android_launcher_binaries
  SOURCE_DIR "${LAUNCHER_SOURCE_DIR}/cmake/android"
  BUILD_COMMAND $(MAKE)
  CMAKE_ARGS
  "-DCMAKE_TOOLCHAIN_FILE=${USE_ANDROID_TOOLCHAIN}"
  "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
  "-DANDROID_ABI=${ANDROID_ABI}"
  "-DCMAKE_CXX_STANDARD=17"
  "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
  "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(android_launcher_binaries BINARY_DIR)
ExternalProject_Add_Step(android_launcher_binaries copy_binaries
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/launcher_android
    ${BINARY_DIR}/libtvm_runtime.so
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDEES install
)

ExternalProject_Add(hexagon_launcher_binaries
  SOURCE_DIR "${LAUNCHER_SOURCE_DIR}/cmake/hexagon"
  BUILD_COMMAND $(MAKE)
  CMAKE_ARGS
  "-DCMAKE_C_COMPILER=${USE_HEXAGON_TOOLCHAIN}/bin/hexagon-clang"
  "-DCMAKE_CXX_COMPILER=${USE_HEXAGON_TOOLCHAIN}/bin/hexagon-clang++"
  "-DCMAKE_CXX_STANDARD=17"
  "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
  "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
  "-DUSE_CUSTOM_LOGGING=ON"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(hexagon_launcher_binaries BINARY_DIR)
ExternalProject_Add_Step(hexagon_launcher_binaries copy_binaries
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/liblauncher_rpc_skel.so
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDEES install
)
