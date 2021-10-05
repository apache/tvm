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
  else(HEXAGON_CLANG)
    message(SEND_ERROR "Cannot find Hexagon toolchain in ${TRY_PATH}")
  endif()
endfunction()

if(BUILD_FOR_HEXAGON)
  find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")
  # Add SDK and QuRT includes when building for Hexagon.
  include_directories(SYSTEM ${HEXAGON_SDK_INCLUDES} ${HEXAGON_QURT_INCLUDES})
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
  file(GLOB RUNTIME_HEXAGON_SIM_SRCS src/runtime/hexagon/android/sim/*.cc)
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
  file(GLOB RUNTIME_HEXAGON_DEVICE_SRCS src/runtime/hexagon/android/target/*.cc)

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

if(BUILD_FOR_HEXAGON AND USE_HEXAGON_DEVICE STREQUAL "${PICK_NONE}")
  file(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/hexagon/*.cc)
else()
  file(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/android/*.cc)
endif()
list(APPEND RUNTIME_SRCS ${RUNTIME_HEXAGON_SRCS} ${RUNTIME_HEXAGON_SIM_SRCS}
                         ${RUNTIME_HEXAGON_DEVICE_SRCS})
