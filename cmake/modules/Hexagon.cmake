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
  tvm_file_glob(GLOB_RECURSE HEXAGON_CLANG "${TRY_PATH}/*/hexagon-clang++")
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

if(BUILD_FOR_HEXAGON)
  find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")
  # Add SDK and QuRT includes when building for Hexagon.
  include_directories(SYSTEM ${HEXAGON_SDK_INCLUDES} ${HEXAGON_QURT_INCLUDES})
endif()

if (NOT USE_HEXAGON_SDK STREQUAL "" AND
    NOT USE_HEXAGON_SDK STREQUAL "/path/to/sdk")
  set(HEXAGON_SDK_PATH_DEFINED ${USE_HEXAGON_SDK})
endif()

if (BUILD_FOR_ANDROID AND HEXAGON_SDK_PATH_DEFINED)
  find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")
  include_directories(SYSTEM
    ${HEXAGON_SDK_INCLUDES}
    ${HEXAGON_RPCMEM_ROOT}/inc
    ${HEXAGON_REMOTE_ROOT})
  link_directories(${HEXAGON_REMOTE_ROOT})
  list(APPEND TVM_RUNTIME_LINKER_LIBS cdsprpc)
endif()

# Don't run these checks when compiling Hexagon device code,
# e.g. when compiling the TVM runtime for Hexagon.
if (NOT BUILD_FOR_HEXAGON AND NOT BUILD_FOR_ANDROID)
  if(USE_HEXAGON_DEVICE STREQUAL "OFF")
    list(APPEND COMPILER_SRCS src/target/opt/build_hexagon_off.cc)
    # append select runtime sources for unit testing
    list(APPEND RUNTIME_SRCS src/runtime/hexagon/hexagon/hexagon_buffer.cc)
    list(APPEND RUNTIME_SRCS src/runtime/hexagon/hexagon/hexagon_common.cc)
    return()
  elseif(NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_SIM}" AND
         NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_HW}")
    message(SEND_ERROR "USE_HEXAGON_DEVICE must be one of "
            "[${PICK_NONE}|${PICK_SIM}|${PICK_HW}]")
    return()
  endif()
endif()

# If no Hexagon support is enabled (other than some stub code), cmake
# execution should stop before reaching this point.

if(NOT USE_HEXAGON_SDK OR NOT USE_HEXAGON_ARCH)
  message(SEND_ERROR "Please set USE_HEXAGON_SDK to the Hexagon SDK root, "
          "and USE_HEXAGON_ARCH to the Hexagon architecture version")
  return()
endif()

if(USE_HEXAGON_LAUNCHER STREQUAL "ON")
  message(SEND_ERROR "USE_HEXAGON_LAUNCHER is deprecated, please build apps separately")
endif()

if(USE_HEXAGON_PROXY_RPC STREQUAL "ON")
  message(SEND_ERROR "USE_HEXAGON_PROXY_RPC is deprecated, please build apps separately")
endif()

# find_hexagon_sdk_root has been called at this point.

if(USE_HEXAGON_RPC)
  set(HEXAGON_RPC_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/hexagon_rpc")
  file(MAKE_DIRECTORY ${HEXAGON_RPC_OUTPUT})

  set(TVMRT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/runtime")
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

  if(BUILD_FOR_ANDROID)
    # Android part
    tvm_file_glob(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/host/*.cc)
    tvm_file_glob(GLOB RUNTIME_HEXAGON_SRCS "${TVMRT_SOURCE_DIR}/hexagon/rpc/android/*.cc")
    list(APPEND RUNTIME_HEXAGON_SRCS "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc_stub.c")

    # copy android_bash template file
    configure_file("${TVMRT_SOURCE_DIR}/hexagon/rpc/android_bash.sh.template"
      ${HEXAGON_RPC_OUTPUT} COPYONLY)

  elseif(BUILD_FOR_HEXAGON)
    # Hexagon part
    find_hexagon_toolchain()
    message(STATUS "HEXAGON_TOOLCHAIN: ${HEXAGON_TOOLCHAIN}")

    add_library(hexagon_rpc_skel SHARED
      "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon_rpc_skel.c"
      "${TVMRT_SOURCE_DIR}/hexagon/rpc/hexagon/rpc_server.cc"
      "${TVMRT_SOURCE_DIR}/minrpc/minrpc_server.h"
      "${TVMRT_SOURCE_DIR}/minrpc/rpc_reference.h"
      "${TVMRT_SOURCE_DIR}/rpc/rpc_module.cc"
      "${TVMRT_SOURCE_DIR}/rpc/rpc_endpoint.cc"
      "${TVMRT_SOURCE_DIR}/rpc/rpc_session.cc"
      "${TVMRT_SOURCE_DIR}/rpc/rpc_local_session.cc"
    )
    target_include_directories(hexagon_rpc_skel
      SYSTEM PRIVATE "${TVMRT_SOURCE_DIR}/hexagon/rpc"
    )
  endif()

  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${HEXAGON_RPC_OUTPUT}")
endif()

if(USE_HEXAGON_DEVICE STREQUAL "${PICK_SIM}")
  find_hexagon_toolchain()
  message(STATUS "Hexagon toolchain: ${HEXAGON_TOOLCHAIN}")
  tvm_file_glob(GLOB RUNTIME_HEXAGON_SIM_SRCS src/runtime/hexagon/android/sim/*.cc)
  include_directories(SYSTEM "${HEXAGON_TOOLCHAIN}/include/iss")
  link_directories("${HEXAGON_TOOLCHAIN}/lib/iss")
  list(APPEND TVM_RUNTIME_LINKER_LIBS "-lwrapper")
  ExternalProject_Add(sim_dev
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/runtime/hexagon/android/sim/driver"
    CMAKE_ARGS
      "-DCMAKE_C_COMPILER=${HEXAGON_TOOLCHAIN}/bin/hexagon-clang"
      "-DCMAKE_CXX_COMPILER=${HEXAGON_TOOLCHAIN}/bin/hexagon-clang++"
      "-DHEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    INSTALL_COMMAND "true"
  )
elseif(USE_HEXAGON_DEVICE STREQUAL "${PICK_HW}")
  find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")
  find_hexagon_toolchain()
  tvm_file_glob(GLOB RUNTIME_HEXAGON_DEVICE_SRCS src/runtime/hexagon/android/target/*.cc)

  include_directories(SYSTEM
    ${HEXAGON_SDK_INCLUDES}
    ${HEXAGON_RPCMEM_ROOT}/inc
    ${HEXAGON_REMOTE_ROOT}
  )

  list(APPEND TVM_RUNTIME_LINKER_LIBS "dl")
  if(BUILD_FOR_ANDROID)
    # Hexagon runtime uses __android_log_print, which is in liblog.
    list(APPEND TVM_RUNTIME_LINKER_LIBS "log")
  endif()
endif()

set(RUNTIME_HEXAGON_COMMON_SRCS src/runtime/hexagon/hexagon_module.cc)
if (USE_HEXAGON_DEVICE STREQUAL "${PICK_NONE}")
  if(BUILD_FOR_HEXAGON)
    tvm_file_glob(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/hexagon/*.cc)
  elseif(BUILD_FOR_ANDROID AND HEXAGON_SDK_PATH_DEFINED)
  else()
    tvm_file_glob(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/host/*.cc)
  endif()
else()
  tvm_file_glob(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/android/*.cc)
endif()

list(APPEND RUNTIME_SRCS
  ${RUNTIME_HEXAGON_SRCS}
  ${RUNTIME_HEXAGON_SIM_SRCS}
  ${RUNTIME_HEXAGON_DEVICE_SRCS}
  ${RUNTIME_HEXAGON_COMMON_SRCS}
)
