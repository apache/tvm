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

if(NOT DEFINED USE_HEXAGON_SDK)
  message(SEND_ERROR "Please set USE_HEXAGON_SDK to the location of Hexagon SDK")
endif()
if (NOT DEFINED USE_HEXAGON_ARCH)
  message(SEND_ERROR "Please set USE_HEXAGON_ARCH to the Hexagon architecture version")
endif()

include(ExternalProject)
include(../../cmake/modules/HexagonSDK.cmake)

find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")

include_directories(SYSTEM ${HEXAGON_SDK_INCLUDES} ${HEXAGON_REMOTE_ROOT})

set(QAIC_EXE "${HEXAGON_QAIC_EXE}")
foreach(INCDIR IN LISTS HEXAGON_SDK_INCLUDES HEXAGON_REMOTE_ROOT)
  list(APPEND QAIC_FLAGS "-I${INCDIR}")
endforeach()

set(LAUNCHER_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_SKIP_RPATH TRUE)

# Qaic for the domain header.
#
# Don't add paths to these filenames, or otherwise cmake may spontaneously
# add -o option to the qaic invocation (with an undesirable path).
set(LAUNCHER_RPC_IDL "launcher_rpc.idl")
set(LAUNCHER_RPC_H "launcher_rpc.h")
set(LAUNCHER_RPC_SKEL_C "launcher_rpc_skel.c")
set(LAUNCHER_RPC_STUB_C "launcher_rpc_stub.c")

add_custom_command(
  OUTPUT ${LAUNCHER_RPC_SKEL_C} ${LAUNCHER_RPC_STUB_C}
    "${LAUNCHER_SRC}/${LAUNCHER_RPC_H}"
  COMMAND ${QAIC_EXE} ${QAIC_FLAGS}
    "${LAUNCHER_SRC}/${LAUNCHER_RPC_IDL}"
  COMMAND ${CMAKE_COMMAND} -E rename "${LAUNCHER_RPC_H}"
    "${LAUNCHER_SRC}/${LAUNCHER_RPC_H}"
  MAIN_DEPENDENCY "${LAUNCHER_SRC}/${LAUNCHER_RPC_IDL}"
)


include_directories(SYSTEM
  "${HEXAGON_SDK_INCLUDES}"
  "${HEXAGON_RPCMEM_ROOT}/inc"
)
include_directories(
  "${LAUNCHER_SRC}"
  "${LAUNCHER_SRC}/../../include"
  "${LAUNCHER_SRC}/../../3rdparty/dlpack/include"
  "${LAUNCHER_SRC}/../../3rdparty/dmlc-core/include"
)
link_directories(${HEXAGON_REMOTE_ROOT})

add_definitions(-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

set(STUB_SRCS
  "launcher_android.cc"
  "launcher_core.cc"
  "launcher_main.cc"
  "launcher_util.cc"
)

add_executable(launcher_android
  "${STUB_SRCS}"
  "${LAUNCHER_RPC_STUB_C}"
)

ExternalProject_Add(android_tvm_runtime
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/../../"
  BUILD_COMMAND $(MAKE) runtime
  CMAKE_ARGS
  "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
  "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
  "-DANDROID_ABI=${ANDROID_ABI}"
  "-DCMAKE_CXX_STANDARD=14"
  "-DUSE_LIBBACKTRACE=OFF"
  "-DUSE_LLVM=OFF"
  "-DUSE_RPC=OFF"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(android_tvm_runtime BINARY_DIR)
ExternalProject_Add_Step(android_tvm_runtime copy_binaries
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${BINARY_DIR} ${CMAKE_INSTALL_PREFIX}
  DEPENDEES install
)

add_dependencies(launcher_android android_tvm_runtime)
add_library(tvm_runtime SHARED IMPORTED)
set_target_properties(tvm_runtime PROPERTIES IMPORTED_LOCATION "${BINARY_DIR}/libtvm_runtime.so")

target_link_libraries(launcher_android cdsprpc log tvm_runtime)
