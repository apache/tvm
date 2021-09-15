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

include(../../../../cmake/modules/HexagonSDK.cmake)

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


if("${FASTRPC_LIBS}" STREQUAL "SKEL")
  # Skel libraries.
  #
  if (NOT DEFINED TVM_RUNTIME_HEXAGON)
    message(FATAL_ERROR "Please set TVM_RUNTIME_HEXAGON=/path/to/libtvm_runtime.a")
  endif()

  include_directories(SYSTEM ${HEXAGON_QURT_INCLUDES})
  include_directories(
    "${LAUNCHER_SRC}"
    "${LAUNCHER_SRC}/../../../../include"
    "${LAUNCHER_SRC}/../../../../3rdparty/dlpack/include"
    "${LAUNCHER_SRC}/../../../../3rdparty/dmlc-core/include"
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

  set(EXTRA_LINK_FLAGS
    "-lposix"
    "-lqurt"
    "-Wl,--export-dynamic"
    "-Wl,--whole-archive ${TVM_RUNTIME_HEXAGON} -Wl,--no-whole-archive"
    "-Wl,--defsym=HEAP_SIZE=0x40000000"
  )
  string(REGEX REPLACE ";" " " EXTRA_LINK_FLAGS_STR "${EXTRA_LINK_FLAGS}")

  set(SKEL_SRCS
    "launcher_core.cc"
    "launcher_hexagon.cc"
  )
  add_library(launcher_rpc_skel SHARED
    "${LAUNCHER_SRC}/${LAUNCHER_RPC_H}"
    "${LAUNCHER_RPC_SKEL_C}"
    "${SKEL_SRCS}"
  )

  # Extra linker flags for linking shared libraries.
  set_target_properties(launcher_rpc_skel PROPERTIES
    LINK_FLAGS ${EXTRA_LINK_FLAGS_STR}
  )
else()
  # Stub libraries.
  #
  if (NOT DEFINED TVM_RUNTIME_ANDROID)
    message(FATAL_ERROR "Please set TVM_RUNTIME_ANDROID=/path/to/libtvm_runtime.so")
  endif()

  include_directories(SYSTEM
    "${HEXAGON_SDK_INCLUDES}"
    "${HEXAGON_RPCMEM_ROOT}/inc"
  )
  include_directories(
    "${LAUNCHER_SRC}"
    "${LAUNCHER_SRC}/../../../../include"
    "${LAUNCHER_SRC}/../../../../3rdparty/dlpack/include"
    "${LAUNCHER_SRC}/../../../../3rdparty/dmlc-core/include"
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
  target_link_libraries(launcher_android cdsprpc log)

  set_target_properties(launcher_android PROPERTIES
    LINK_FLAGS "${TVM_RUNTIME_ANDROID}"
  )
endif()
