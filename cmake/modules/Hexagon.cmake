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

set(PICK_SIM  "sim")
set(PICK_HW   "target")
set(PICK_NONE "OFF")

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

set(TVMRT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/runtime")

# First, verify that USE_HEXAGON_DEVICE has a valid value.
if(DEFINED USE_HEXAGON_DEVICE)
  if(NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_SIM}" AND
     NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_HW}" AND
     NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_NONE}")
    message(SEND_ERROR "USE_HEXAGON_DEVICE must be one of "
            "[${PICK_NONE}|${PICK_SIM}|${PICK_HW}]")
    set(USE_HEXAGON_DEVICE OFF)
  endif()
endif()

# This .cmake file is included when building any part of TVM for any
# architecture. It shouldn't require any Hexagon-specific parameters
# (like the path to the SDK), unless it's needed.
#
# Aside from building the code for Hexagon, two flags can enable some
# Hexagon-related functionality:
# - USE_HEXAGON_DEVICE
# - USE_HEXAGON_RPC
#
# USE_HEXAGON_RPC:
# - When building for Hexagon, this will build the Hexagon endpoint of the
#   RPC server: the FastRPC skel library (with TVM runtime built into it).
# - When building for Android, this will build the (intermediary) RPC server,
#   including the "stub" code for the FastRPC implementation of the RPC
#   channel.

if(NOT BUILD_FOR_HEXAGON AND NOT BUILD_FOR_ANDROID)
  set(BUILD_FOR_HOST TRUE)
endif()


if(NOT USE_HEXAGON_DEVICE AND NOT USE_HEXAGON_RPC AND NOT BUILD_FOR_HEXAGON)
  # If nothing related to Hexagon is enabled, add phony Hexagon codegen,
  # and some stuff needed by cpptests (this part is a temporary workaround
  # until e2e support for Hexagon is enabled).
  if(BUILD_FOR_HOST)
    list(APPEND COMPILER_SRCS src/target/opt/build_hexagon_off.cc)
  endif()
  list(APPEND RUNTIME_SRCS src/runtime/hexagon/hexagon/hexagon_buffer.cc)
  list(APPEND RUNTIME_SRCS src/runtime/hexagon/hexagon/hexagon_common.cc)
  list(APPEND RUNTIME_SRCS src/runtime/hexagon/hexagon/hexagon_user_dma.cc)
  return()
endif()


function(add_android_paths)
  if(NOT DEFINED HEXAGON_SDK_INCLUDES OR
     NOT DEFINED HEXAGON_RPCMEM_ROOT OR
     NOT DEFINED HEXAGON_REMOTE_ROOT)
    message(FATAL_ERROR "This function must be called after find_hexagon_sdk_root")
  endif()
  include_directories(SYSTEM
    ${HEXAGON_SDK_INCLUDES}
    ${HEXAGON_RPCMEM_ROOT}/inc
    ${HEXAGON_REMOTE_ROOT}
  )
  link_directories(${HEXAGON_REMOTE_ROOT})
endfunction()


# Common sources for TVM runtime with Hexagon support
file_glob_append(RUNTIME_HEXAGON_COMMON_SRCS
  "${TVMRT_SOURCE_DIR}/hexagon/hexagon_module.cc"
  "${TVMRT_SOURCE_DIR}/hexagon/hexagon/*.cc"
)


if(BUILD_FOR_HEXAGON)
  find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")
  # Add SDK and QuRT includes when building for Hexagon.
  include_directories(SYSTEM ${HEXAGON_SDK_INCLUDES} ${HEXAGON_QURT_INCLUDES})

  list(APPEND RUNTIME_HEXAGON_SRCS ${RUNTIME_HEXAGON_COMMON_SRCS})
endif()


if(USE_HEXAGON_DEVICE)
  function(invalid_device_value_for BUILD_TARGET)
    message(SEND_ERROR
      "USE_HEXAGON_DEVICE=${USE_HEXAGON_DEVICE} is not supported when "
      "building for ${BUILD_TARGET}"
    )
  endfunction()

  list(APPEND RUNTIME_HEXAGON_SRCS ${RUNTIME_HEXAGON_COMMON_SRCS})

  if(BUILD_FOR_HOST)
    if(NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_SIM}")
      invalid_device_value_for("host")
    endif()
    find_hexagon_toolchain()
    file_glob_append(RUNTIME_HEXAGON_SRCS
      "${TVMRT_SOURCE_DIR}/hexagon/android/*.cc"
      "${TVMRT_SOURCE_DIR}/hexagon/android/sim/*.cc"
    )
    include_directories(SYSTEM "${HEXAGON_TOOLCHAIN}/include/iss")
    link_directories("${HEXAGON_TOOLCHAIN}/lib/iss")
    list(APPEND TVM_RUNTIME_LINKER_LIBS "-lwrapper")

    ExternalProject_Add(sim_dev
      SOURCE_DIR "${TVMRT_SOURCE_DIR}/hexagon/android/sim/driver"
      CMAKE_ARGS
        "-DCMAKE_C_COMPILER=${HEXAGON_TOOLCHAIN}/bin/hexagon-clang"
        "-DCMAKE_CXX_COMPILER=${HEXAGON_TOOLCHAIN}/bin/hexagon-clang++"
        "-DHEXAGON_ARCH=${USE_HEXAGON_ARCH}"
      INSTALL_COMMAND "true"
    )

  elseif(BUILD_FOR_ANDROID)
    if(NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_HW}")
      invalid_device_value_for("Android")
    endif()
    find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")
    find_hexagon_toolchain()
    add_android_paths()
    file_glob_append(RUNTIME_HEXAGON_SRCS
      "${TVMRT_SOURCE_DIR}/hexagon/android/*.cc"
      "${TVMRT_SOURCE_DIR}/hexagon/android/target/*.cc"
    )
    # Hexagon runtime uses __android_log_print, which is in liblog.
    list(APPEND TVM_RUNTIME_LINKER_LIBS dl log cdsprpc)

  elseif(BUILD_FOR_HEXAGON)
    invalid_device_value_for("Hexagon")
  endif()
endif()   # USE_HEXAGON_DEVICE


if(USE_HEXAGON_RPC)
  function(build_rpc_idl)
    set(QAIC_EXE "${HEXAGON_QAIC_EXE}")
    foreach(INCDIR IN LISTS HEXAGON_SDK_INCLUDES HEXAGON_REMOTE_ROOT)
      list(APPEND QAIC_FLAGS "-I${INCDIR}")
    endforeach()

    add_custom_command(
      OUTPUT
        "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc.h"
        "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc_skel.c"
        "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc_stub.c"
      COMMAND
        ${QAIC_EXE} ${QAIC_FLAGS} "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc.idl"
          -o "${TVMRT_SOURCE_DIR}/hexagon/rpc"
      MAIN_DEPENDENCY "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc.idl"
    )
  endfunction()

  list(APPEND RUNTIME_HEXAGON_SRCS ${RUNTIME_HEXAGON_COMMON_SRCS})

  if(BUILD_FOR_ANDROID)
    # Android part
    find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")
    add_android_paths()
    build_rpc_idl()
    file_glob_append(RUNTIME_HEXAGON_SRCS
      "${TVMRT_SOURCE_DIR}/hexagon/host/*.cc"
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
    find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")
    find_hexagon_toolchain()
    build_rpc_idl()

    # Include the generic RPC code into the TVM runtime.
    list(APPEND RUNTIME_HEXAGON_SRCS
      "${TVMRT_SOURCE_DIR}/minrpc/minrpc_server.h"
      "${TVMRT_SOURCE_DIR}/minrpc/rpc_reference.h"
      "${TVMRT_SOURCE_DIR}/rpc/rpc_module.cc"
      "${TVMRT_SOURCE_DIR}/rpc/rpc_endpoint.cc"
      "${TVMRT_SOURCE_DIR}/rpc/rpc_session.cc"
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
  endif()
endif()   # USE_HEXAGON_RPC


list(APPEND RUNTIME_SRCS ${RUNTIME_HEXAGON_SRCS})
