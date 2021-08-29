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

set(PICK_SIM  "sim")
set(PICK_HW   "target")
set(PICK_NONE "OFF")

set(FOUND_HEXAGON_SDK_ROOT  FALSE)
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
  else(HEXAGON_CLANG)
    message(SEND_ERROR "Cannot find Hexagon toolchain in ${TRY_PATH}")
  endif()
endfunction()

function(find_hexagon_sdk_root)
  if(FOUND_HEXAGON_SDK_ROOT)
    return()
  endif()
  message(STATUS "Checking Hexagon SDK root: ${USE_HEXAGON_SDK}")
  file(GLOB_RECURSE HEXAGON_AEESTDDEF "${USE_HEXAGON_SDK}/*/AEEStdDef.h")
  if(HEXAGON_AEESTDDEF)
    # The path is ${HEXAGON_SDK_ROOT}/incs/stddef/AEEStdDef.h.
    get_filename_component(HEXAGON_TMP0 "${HEXAGON_AEESTDDEF}" DIRECTORY)
    get_filename_component(HEXAGON_TMP1 "${HEXAGON_TMP0}" DIRECTORY)
    get_filename_component(HEXAGON_TMP2 "${HEXAGON_TMP1}" DIRECTORY)
    set(HEXAGON_SDK_ROOT "${HEXAGON_TMP2}" CACHE PATH
        "Root directory of Hexagon SDK")
    set(FOUND_HEXAGON_SDK_ROOT TRUE)
  else(HEXAGON_AEESTDDEF)
    message(SEND_ERROR "Cannot validate Hexagon SDK in ${USE_HEXAGON_SDK}")
  endif()
endfunction()

if(BUILD_FOR_HEXAGON)
  find_hexagon_sdk_root()
  if(HEXAGON_SDK_ROOT MATCHES "3.5.1")
    message(SEND_ERROR "Hexagon SDK 3.5.1 is not supported")
  elseif(HEXAGON_SDK_ROOT MATCHES "3\.[0-9]+\.[0-9]+")
    include_directories(
      SYSTEM "${USE_HEXAGON_SDK}/libs/common/qurt/ADSPv62MP/include/posix"
      SYSTEM "${USE_HEXAGON_SDK}/libs/common/qurt/ADSPv62MP/include/qurt")
  else()
    include_directories(
      SYSTEM "${HEXAGON_SDK_ROOT}/rtos/qurt/computev65/include/posix"
      SYSTEM "${HEXAGON_SDK_ROOT}/rtos/qurt/computev65/include/qurt")
  endif()
  include_directories(
    SYSTEM "${HEXAGON_SDK_ROOT}/incs"
    SYSTEM "${HEXAGON_SDK_ROOT}/incs/stddef")
endif()

if(USE_HEXAGON_DEVICE STREQUAL "OFF")
  list(APPEND COMPILER_SRCS src/target/opt/build_hexagon_off.cc)
  return()
elseif(NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_SIM}" AND
       NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_HW}")
  set(ERROR_MSG
      "USE_HEXAGON_DEVICE must be one of [${PICK_NONE}|${PICK_SIM}|${PICK_HW}]")
  message(SEND_ERROR "${ERROR_MSG}")
  return()
endif()
# If USE_HEXAGON_DEVICE is set to a valid value, make sure that USE_HEXAGON_SDK
# is defined.
if(NOT USE_HEXAGON_SDK)
  message(SEND_ERROR "Please set USE_HEXAGON_SDK to the Hexagon SDK root")
  return()
endif()

if(USE_HEXAGON_DEVICE STREQUAL "${PICK_SIM}")
  find_hexagon_toolchain()
  message(STATUS "Hexagon toolchain: ${HEXAGON_TOOLCHAIN}")
  file(GLOB RUNTIME_HEXAGON_SIM_SRCS src/runtime/hexagon/sim/*.cc)
  include_directories(SYSTEM "${HEXAGON_TOOLCHAIN}/include/iss")
  link_directories("${HEXAGON_TOOLCHAIN}/lib/iss")
  list(APPEND TVM_RUNTIME_LINKER_LIBS "-lwrapper")
  ExternalProject_Add(sim_dev
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/runtime/hexagon/sim/driver"
    CMAKE_ARGS
      "-DCMAKE_C_COMPILER=${HEXAGON_TOOLCHAIN}/bin/hexagon-clang"
      "-DCMAKE_CXX_COMPILER=${HEXAGON_TOOLCHAIN}/bin/hexagon-clang++"
    INSTALL_COMMAND "true"
  )
elseif(USE_HEXAGON_DEVICE STREQUAL "${PICK_HW}")
  find_hexagon_sdk_root()
  find_hexagon_toolchain()
  message(STATUS "Hexagon SDK: ${HEXAGON_SDK_ROOT}")
  if(HEXAGON_SDK_ROOT MATCHES "3.5.1")
    message(SEND_ERROR "Hexagon SDK 3.5.1 is not supported")
  elseif(HEXAGON_SDK_ROOT MATCHES "3\.[0-9]+\.[0-9]+")
      set(RPCMEM_DIR "libs/common/rpcmem")
      set(REMOTE_DIR "libs/common/remote/ship/android_Release_aarch64")
  else()
      set(RPCMEM_DIR "ipc/fastrpc/rpcmem")
      set(REMOTE_DIR "ipc/fastrpc/remote/ship/android_aarch64")
  endif()
  file(GLOB RUNTIME_HEXAGON_DEVICE_SRCS src/runtime/hexagon/target/*.cc)
  include_directories(SYSTEM "${HEXAGON_SDK_ROOT}/incs/stddef")
  include_directories(SYSTEM "${HEXAGON_SDK_ROOT}/${RPCMEM_DIR}/inc")
  include_directories(
      SYSTEM "${HEXAGON_SDK_ROOT}/incs")
  include_directories(
      SYSTEM "${HEXAGON_SDK_ROOT}/${REMOTE_DIR}")
  include_directories(SYSTEM "${HEXAGON_TOOLCHAIN}/include/iss")
  list(APPEND TVM_RUNTIME_LINKER_LIBS "dl")
  if(BUILD_FOR_ANDROID)
    # Hexagon runtime uses __android_log_print, which is in liblog.
    list(APPEND TVM_RUNTIME_LINKER_LIBS "log")
  endif()
endif()

file(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/*.cc)
list(APPEND RUNTIME_SRCS ${RUNTIME_HEXAGON_SRCS} ${RUNTIME_HEXAGON_SIM_SRCS}
                         ${RUNTIME_HEXAGON_DEVICE_SRCS})

