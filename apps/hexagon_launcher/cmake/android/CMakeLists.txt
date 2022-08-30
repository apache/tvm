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
project(HexagonAndroidLauncher C CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/../HexagonLauncher.cmake")
# From the include above, get
#   SDK_INCLUDE_DIRS
#   QAIC_EXE_PATH
# and
#   QAIC_FLAGS
#   LAUNCHER_SRC
#   LAUNCHER_RPC_IDL
#   LAUNCHER_RPC_H
#   LAUNCHER_RPC_SKEL_C
#   LAUNCHER_RPC_STUB_C

add_custom_command(
  OUTPUT ${LAUNCHER_RPC_STUB_C} ${LAUNCHER_RPC_H}
  COMMAND ${QAIC_EXE_PATH} ${QAIC_FLAGS} "${LAUNCHER_SRC}/${LAUNCHER_RPC_IDL}"
  MAIN_DEPENDENCY "${LAUNCHER_SRC}/${LAUNCHER_RPC_IDL}"
)

get_hexagon_sdk_property("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}"
  RPCMEM_ROOT RPCMEM_ROOT_DIR
  DSPRPC_LIB  DSPRPC_LIB_DIRS
)
if(NOT RPCMEM_ROOT_DIR)
  message(WARNING "Could not locate some Hexagon SDK components")
endif()

include_directories(SYSTEM
  "${SDK_INCLUDE_DIRS}"
  "${RPCMEM_ROOT_DIR}/inc"
  "${CMAKE_CURRENT_BINARY_DIR}"   # Output of qaic will go here
)

link_directories(${DSPRPC_LIB_DIRS})

add_definitions(-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

set(STUB_SRCS
  "${LAUNCHER_SRC}/launcher_android.cc"
  "${LAUNCHER_SRC}/launcher_core.cc"
  "${LAUNCHER_SRC}/launcher_main.cc"
  "${LAUNCHER_SRC}/launcher_util.cc"
)

add_executable(launcher_android
  "${LAUNCHER_RPC_H}"
  "${LAUNCHER_RPC_STUB_C}"
  "${STUB_SRCS}"
)

ExternalProject_Add(android_tvm_runtime
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND $(MAKE) runtime
  CMAKE_ARGS
  "-DANDROID_ABI=${ANDROID_ABI}"
  "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
  "-DCMAKE_CXX_STANDARD=17"
  "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
  "-DUSE_HEXAGON=ON"
  "-DUSE_GRAPH_EXECUTOR=OFF"
  "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
  "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
  "-DUSE_LIBBACKTRACE=OFF"
  "-DUSE_LLVM=OFF"
  "-DUSE_PROFILER=OFF"
  "-DUSE_RPC=OFF"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(android_tvm_runtime BINARY_DIR)
ExternalProject_Add_Step(android_tvm_runtime copy_binaries
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/libtvm_runtime.so
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDEES install
)

add_dependencies(launcher_android android_tvm_runtime)
add_library(a_tvm_runtime SHARED IMPORTED)
set_target_properties(a_tvm_runtime PROPERTIES IMPORTED_LOCATION "${BINARY_DIR}/libtvm_runtime.so")

target_link_libraries(launcher_android cdsprpc log a_tvm_runtime)
