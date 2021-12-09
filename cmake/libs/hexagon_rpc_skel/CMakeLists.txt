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
include(ExternalProject)
project(HexagonRPCSkel C CXX)

set(TVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(TVM_SRC_DIR "${TVM_SOURCE_DIR}/src")


include("${TVM_SOURCE_DIR}/cmake/modules/HexagonSDK.cmake")

find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")

include_directories(SYSTEM ${HEXAGON_SDK_INCLUDES} ${HEXAGON_REMOTE_ROOT})

set(HEXAGON_RPC_H "hexagon_rpc.h")
set(HEXAGON_RPC_SKEL_C "hexagon_rpc_skel.c")
set(HEXAGON_RPC_STUB_C "hexagon_rpc_stub.c")

include_directories(
  "${TVM_SOURCE_DIR}/include"
  "${TVM_SOURCE_DIR}/3rdparty/dlpack/include"
  "${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include"
)

set(QAIC_EXE "${HEXAGON_QAIC_EXE}")
foreach(INCDIR IN LISTS HEXAGON_SDK_INCLUDES HEXAGON_REMOTE_ROOT)
  list(APPEND QAIC_FLAGS "-I${INCDIR}")
endforeach()

add_custom_command(
  OUTPUT ${HEXAGON_RPC_SKEL_C} ${HEXAGON_RPC_H}
  COMMAND ${QAIC_EXE} ${QAIC_FLAGS} "${TVM_SRC_DIR}/runtime/hexagon/rpc/hexagon_rpc.idl"
  MAIN_DEPENDENCY "${TVM_SRC_DIR}/runtime/hexagon/rpc/hexagon_rpc.idl"
)

include_directories(SYSTEM
  ${HEXAGON_QURT_INCLUDES}
  ${CMAKE_CURRENT_BINARY_DIR}   # Output of qaic will go here
)

link_directories(${HEXAGON_QURT_LIBS})

add_definitions(-D_MACH_I32=int)
add_definitions(-DDMLC_CXX11_THREAD_LOCAL=0)
add_definitions(-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

# Extra compile flags (both C and C++).
set(EXTRA_COMP_FLAGS
  "-O3"
  "-m${USE_HEXAGON_ARCH}"
)
string(REGEX REPLACE ";" " " EXTRA_COMP_FLAGS_STR "${EXTRA_COMP_FLAGS}")
set(CMAKE_C_FLAGS "${EXTRA_COMP_FLAGS_STR} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${EXTRA_COMP_FLAGS_STR} ${CMAKE_CXX_FLAGS}")

set(SKEL_SRCS
  "${TVM_SRC_DIR}/runtime/hexagon/rpc/hexagon/rpc_server.cc"
)

set(MINRPC_SRCS
  "${TVM_SRC_DIR}/runtime/minrpc/minrpc_server.h"
  "${TVM_SRC_DIR}/runtime/minrpc/rpc_reference.h"
)

set(TVM_RPC_SRC 
  "${TVM_SRC_DIR}/runtime/rpc/rpc_module.cc"
  "${TVM_SRC_DIR}/runtime/rpc/rpc_endpoint.cc"
  "${TVM_SRC_DIR}/runtime/rpc/rpc_session.cc"
  "${TVM_SRC_DIR}/runtime/rpc/rpc_local_session.cc"
)

add_library(hexagon_rpc_skel SHARED
  "${HEXAGON_RPC_H}"
  "${HEXAGON_RPC_SKEL_C}"
  "${SKEL_SRCS}"
  "${MINRPC_SRCS}"
  "${TVM_RPC_SRC}"
)

ExternalProject_Add(static_hexagon_tvm_runtime
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND $(MAKE) runtime
  CMAKE_ARGS
  "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
  "-DCMAKE_CXX_STANDARD=14"
  "-DUSE_LIBBACKTRACE=OFF"
  "-DUSE_LLVM=OFF"
  "-DUSE_RPC=OFF"
  "-DBUILD_STATIC_RUNTIME=ON"
  "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(static_hexagon_tvm_runtime BINARY_DIR)

add_dependencies(hexagon_rpc_skel static_hexagon_tvm_runtime)
add_library(h_tvm_runtime STATIC IMPORTED)
set_target_properties(h_tvm_runtime PROPERTIES IMPORTED_LOCATION "${BINARY_DIR}/libtvm_runtime.a")

target_link_libraries(hexagon_rpc_skel -Wl,--whole-archive h_tvm_runtime -Wl,--no-whole-archive)
