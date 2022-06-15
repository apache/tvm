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

include(ExternalProject)
include(cmake/modules/HexagonSDK.cmake)

set(FOUND_HEXAGON_TOOLCHAIN FALSE)

function(find_hexagon_toolchain)
  if(FOUND_HEXAGON_TOOLCHAIN)
    return()
  endif()
  if(NOT "${USE_HEXAGON_TOOLCHAIN}" STREQUAL "")
    set(TRY_PATH "${USE_HEXAGON_TOOLCHAIN}")
  else()
    set(TRY_PATH "${USE_HEXAGON_SDK}")
  endif()
  message(STATUS "Looking for Hexagon toolchain in ${TRY_PATH}")
  file(GLOB_RECURSE HEXAGON_CLANG "${TRY_PATH}/*/hexagon-clang++")
  if(HEXAGON_CLANG)
    # The path is ${HEXAGON_TOOLCHAIN}/bin/hexagon-clang++.
    get_filename_component(HEXAGON_TMP0 "${HEXAGON_CLANG}" DIRECTORY)
    get_filename_component(HEXAGON_TMP1 "${HEXAGON_TMP0}" DIRECTORY)
    set(HEXAGON_TOOLCHAIN "${HEXAGON_TMP1}" CACHE PATH
        "Path to the Hexagon toolchain")
    set(FOUND_HEXAGON_TOOLCHAIN TRUE)
  else()
    message(SEND_ERROR "Cannot find Hexagon toolchain in ${TRY_PATH}")
  endif()
endfunction()

macro(file_glob_append _output_list)
  tvm_file_glob(GLOB _tmp0 ${ARGN})
  set(_tmp1 ${${_output_list}})
  list(APPEND _tmp1 ${_tmp0})
  set(${_output_list} ${_tmp1})
endmacro()

set(TVMRT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime")

if(DEFINED USE_HEXAGON_DEVICE)
  message(WARNING "USE_HEXAGON_DEVICE is deprecated, use USE_HEXAGON instead")
endif()

# This .cmake file is included when building any part of TVM for any
# architecture. It shouldn't require any Hexagon-specific parameters (like
# the path to the SDK), unless it's needed. The flag USE_HEXAGON decides
# whether any Hexagon-related functionality is enabled. Specifically,
# setting USE_HEXAGON=OFF, disables any form of Hexagon support.
#
# Note on the function of USE_HEXAGON_RPC:
# - When building for Hexagon, this will build the Hexagon endpoint of the
#   RPC server: the FastRPC skel library (with TVM runtime built into it),
#   and the standalone RPC server for simulator.
# - When building for Android, this will build the (intermediary) RPC server,
#   including the "stub" code for the FastRPC implementation of the RPC
#   channel.
# - When building for x86, this will build the host-side code that instan-
#   tiates the simulator.

if(NOT BUILD_FOR_HEXAGON AND NOT BUILD_FOR_ANDROID)
  set(BUILD_FOR_HOST TRUE)
endif()


if(NOT USE_HEXAGON)
  # If nothing related to Hexagon is enabled, add phony Hexagon codegen,
  # and some stuff needed by cpptests (this part is a temporary workaround
  # until e2e support for Hexagon is enabled).
  if(BUILD_FOR_HOST)
    list(APPEND COMPILER_SRCS src/target/opt/build_hexagon_off.cc)
  endif()
  return()
endif()

# From here on, USE_HEXAGON is assumed to be TRUE.

function(add_android_paths)
  get_hexagon_sdk_property("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}"
    SDK_INCLUDE SDK_INCLUDE_DIRS
    DSPRPC_LIB  DSPRPC_LIB_DIRS
    RPCMEM_ROOT RPCMEM_ROOT_DIR
  )
  if(NOT SDK_INCLUDE_DIRS OR NOT DSPRPC_LIB_DIRS OR NOT RPCMEM_ROOT_DIR)
    message(WARNING "Could not locate some Hexagon SDK components")
  endif()

  include_directories(SYSTEM
    ${SDK_INCLUDE_DIRS}
    "${RPCMEM_ROOT_DIR}/inc"
  )
  link_directories(${DSPRPC_LIB_DIRS})
endfunction()

function(add_hexagon_wrapper_paths)
  if(NOT DEFINED HEXAGON_TOOLCHAIN)
    message(FATAL_ERROR "This function must be called after find_hexagon_toolchain")
  endif()
  include_directories(SYSTEM
    "${HEXAGON_TOOLCHAIN}/include/iss"
  )
  link_directories("${HEXAGON_TOOLCHAIN}/lib/iss")
endfunction()

if(BUILD_FOR_HEXAGON)
  # Common sources for TVM runtime with Hexagon support
  file_glob_append(RUNTIME_HEXAGON_SRCS
    "${TVMRT_SOURCE_DIR}/hexagon/*.cc"
  )
else()
  file_glob_append(RUNTIME_HEXAGON_SRCS
    "${TVMRT_SOURCE_DIR}/hexagon/hexagon_module.cc"
  )
endif()

if(BUILD_FOR_HEXAGON)
  if(DEFINED USE_HEXAGON_GTEST AND EXISTS ${USE_HEXAGON_GTEST})
    file_glob_append(RUNTIME_HEXAGON_SRCS
      "${CMAKE_SOURCE_DIR}/tests/cpp-runtime/hexagon/*.cc"
    )
  endif()
  get_hexagon_sdk_property("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}"
    SDK_INCLUDE   SDK_INCLUDE_DIRS
    QURT_INCLUDE  QURT_INCLUDE_DIRS
  )
  if(NOT SDK_INCLUDE_DIRS OR NOT QURT_INCLUDE_DIRS)
    message(WARNING "Could not locate some Hexagon SDK components")
  endif()

  # Set the compiler arch flag.
  add_definitions("-m${USE_HEXAGON_ARCH}")

  # Add SDK and QuRT includes when building for Hexagon.
  include_directories(SYSTEM ${SDK_INCLUDE_DIRS} ${QURT_INCLUDE_DIRS})

  set(USE_CUSTOM_LOGGING ON) # To use a custom logger
endif()


if(USE_HEXAGON_RPC)
  function(build_rpc_idl)
    get_hexagon_sdk_property("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}"
      SDK_INCLUDE   SDK_INCLUDE_DIRS
      QAIC_EXE      QAIC_EXE_PATH
    )
    foreach(INCDIR IN LISTS SDK_INCLUDE_DIRS)
      list(APPEND QAIC_FLAGS "-I${INCDIR}")
    endforeach()

    add_custom_command(
      OUTPUT
        "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc.h"
        "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc_skel.c"
        "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc_stub.c"
      COMMAND
        ${QAIC_EXE_PATH} ${QAIC_FLAGS}
          "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc.idl"
          -o "${TVMRT_SOURCE_DIR}/hexagon/rpc"
      MAIN_DEPENDENCY "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc.idl"
    )

    if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
        # We can't easily fix this at the source-code level, because the .c file is generated
        # by the qaic program.  But it should be safe to ignore the warning:
        # https://stackoverflow.com/questions/13905200/is-it-wise-to-ignore-gcc-clangs-wmissing-braces-warning
        set_source_files_properties("${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc_stub.c"
            PROPERTY COMPILE_FLAGS "-Wno-missing-braces")
    endif()
  endfunction()

  if(BUILD_FOR_ANDROID)
    # Android part
    add_android_paths()
    build_rpc_idl()
    file_glob_append(RUNTIME_HEXAGON_SRCS
      "${TVMRT_SOURCE_DIR}/hexagon/rpc/android/*.cc"
    )
    # Add this file separately, because it's auto-generated, and glob won't
    # find it during cmake-time.
    list(APPEND RUNTIME_HEXAGON_SRCS
      "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc_stub.c"
    )
    list(APPEND TVM_RUNTIME_LINKER_LIBS cdsprpc)

  elseif(BUILD_FOR_HEXAGON)
    # Hexagon part
    find_hexagon_toolchain()
    build_rpc_idl()

    # Include the generic RPC code into the TVM runtime.
    list(APPEND RUNTIME_HEXAGON_SRCS
      "${TVMRT_SOURCE_DIR}/minrpc/minrpc_server.h"
      "${TVMRT_SOURCE_DIR}/minrpc/rpc_reference.h"
      "${TVMRT_SOURCE_DIR}/rpc/rpc_module.cc"
      "${TVMRT_SOURCE_DIR}/rpc/rpc_endpoint.cc"
      "${TVMRT_SOURCE_DIR}/rpc/rpc_session.cc"
      # TODO(masahi): Remove rpc_local_session.cc after verifying that things work without it
      "${TVMRT_SOURCE_DIR}/rpc/rpc_local_session.cc"
    )
    # Add the hardware-specific RPC code into the skel library.
    add_library(hexagon_rpc_skel SHARED
      "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon/rpc_server.cc"
      "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc_skel.c"
    )
    target_include_directories(hexagon_rpc_skel
      SYSTEM PRIVATE "${TVMRT_SOURCE_DIR}/hexagon/rpc"
    )
    # Add the simulator-specific RPC code into a shared library to be
    # executed via run_main_on_sim.
    add_library(hexagon_rpc_sim SHARED
      "${TVMRT_SOURCE_DIR}/hexagon/rpc/simulator/rpc_server.cc"
    )
    target_link_libraries(hexagon_rpc_sim
      -Wl,--whole-archive tvm_runtime -Wl,--no-whole-archive
    )

  elseif(BUILD_FOR_HOST)
    find_hexagon_toolchain()
    add_hexagon_wrapper_paths()
    file_glob_append(RUNTIME_HEXAGON_SRCS
      "${TVMRT_SOURCE_DIR}/hexagon/rpc/simulator/session.cc"
    )
    list(APPEND TVM_RUNTIME_LINKER_LIBS "-lwrapper")
  endif()
endif()   # USE_HEXAGON_RPC


list(APPEND RUNTIME_SRCS ${RUNTIME_HEXAGON_SRCS})
