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

if(NOT "${FASTRPC_LIBS}" STREQUAL "SKEL" AND
   NOT "${FASTRPC_LIBS}" STREQUAL "STUB")
  message(SEND_ERROR "Please set FASTRPC_LIBS to either SKEL or STUB")
endif()

if(NOT DEFINED USE_HEXAGON_SDK)
  message(SEND_ERROR "Please set USE_HEXAGON_SDK to the location of Hexagon SDK")
endif()
if (NOT DEFINED USE_HEXAGON_ARCH)
  message(SEND_ERROR "Please set USE_HEXAGON_ARCH to the Hexagon architecture version")
endif()

include(ExternalProject)
include(../../../cmake/modules/HexagonSDK.cmake)

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
set(LAUNCHER_RPC_IDL "../launcher_rpc.idl")
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


include_directories(SYSTEM ${HEXAGON_QURT_INCLUDES})
include_directories(
  "${LAUNCHER_SRC}"
  "${LAUNCHER_SRC}/../../../include"
  "${LAUNCHER_SRC}/../../../3rdparty/dlpack/include"
  "${LAUNCHER_SRC}/../../../3rdparty/dmlc-core/include"
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
  "../launcher_core.cc"
  "../launcher_hexagon.cc"
)
add_library(launcher_rpc_skel SHARED
  "${LAUNCHER_SRC}/${LAUNCHER_RPC_H}"
  "${LAUNCHER_RPC_SKEL_C}"
  "${SKEL_SRCS}"
)


ExternalProject_Add(static_hexagon_tvm_runtime
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/../../../"
  BUILD_COMMAND $(MAKE) runtime
  CMAKE_ARGS
  "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DUSE_HEXAGON_ARCH=v68"
  "-DCMAKE_CXX_STANDARD=14"
  "-DUSE_LIBBACKTRACE=OFF"
  "-DUSE_LLVM=OFF"
  "-DUSE_RPC=OFF"
  "-DBUILD_STATIC_RUNTIME=ON"
  "-DUSE_HEXAGON_SDK=/opt/qualcomm/hexagon/SDK/4.2.0.2"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(static_hexagon_tvm_runtime BINARY_DIR)
ExternalProject_Add_Step(static_hexagon_tvm_runtime copy_binaries
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${BINARY_DIR} ${CMAKE_INSTALL_PREFIX}
  DEPENDEES install
)

add_dependencies(launcher_rpc_skel static_hexagon_tvm_runtime)
add_library(static_tvm_runtime STATIC IMPORTED)
set_target_properties(static_tvm_runtime PROPERTIES IMPORTED_LOCATION "${BINARY_DIR}/libtvm_runtime.a")

target_link_libraries(launcher_rpc_skel -Wl,--whole-archive static_tvm_runtime -Wl,--no-whole-archive)

