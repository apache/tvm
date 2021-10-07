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

if(USE_HEXAGON_LAUNCHER STREQUAL "ON")
  set(USE_HEXAGON_DEVICE "${PICK_SIM}")
else()
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
endif()

# If USE_HEXAGON_DEVICE/LAUNCHER is set to a valid value, make sure that USE_HEXAGON_SDK
# is defined.
if(NOT USE_HEXAGON_SDK)
  message(SEND_ERROR "Please set USE_HEXAGON_SDK to the Hexagon SDK root")
  return()
endif()

if(USE_HEXAGON_LAUNCHER STREQUAL "ON")

  if(DEFINED USE_ANDROID_TOOLCHAIN)
    if(NOT DEFINED ANDROID_PLATFORM)
      message(SEND_ERROR "Please set ANDROID_PLATFORM "
        "when providing an Android cmake toolchain.")
    endif()
    if(NOT DEFINED ANDROID_ABI)
      message(SEND_ERROR "Please set ANDROID_ABI "
        "when providing an Android cmake toolchain.")
    endif()
  else()
    message(SEND_ERROR "Please set USE_ANDROID_TOOLCHAIN to build the android "
      " launcher for hexagon.")
  endif()

  set(LAUNCHER_BINARY_DIR "${CMAKE_BINARY_DIR}/launcher")
  ExternalProject_Add(launcher_android
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/apps/hexagon_launcher"
    INSTALL_DIR "${LAUNCHER_BINARY_DIR}"
    BUILD_ALWAYS ON
    CMAKE_ARGS
    "-DCMAKE_TOOLCHAIN_FILE=${USE_ANDROID_TOOLCHAIN}"
    "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
    "-DANDROID_ABI=${ANDROID_ABI}"
    "-DFASTRPC_LIBS=STUB"
    "-DUSE_HEXAGON_ARCH=v68"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    "-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(launcher_android BINARY_DIR)
  ExternalProject_Add_Step(launcher_android copy_binaries
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${BINARY_DIR} ${LAUNCHER_BINARY_DIR}
    DEPENDEES install
  )
  ExternalProject_Add(launcher_hexagon
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/apps/hexagon_launcher/rpc_skel"
    INSTALL_DIR "${LAUNCHER_BINARY_DIR}"
    BUILD_ALWAYS ON
    CMAKE_ARGS
    "-DCMAKE_C_COMPILER=${USE_HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang"
    "-DCMAKE_CXX_COMPILER=${USE_HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang++"
    "-DFASTRPC_LIBS=SKEL"
    "-DUSE_HEXAGON_ARCH=v68"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    "-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(launcher_hexagon BINARY_DIR)
  ExternalProject_Add_Step(launcher_hexagon copy_binaries
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${BINARY_DIR} ${LAUNCHER_BINARY_DIR}
    DEPENDEES install
  )

  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${LAUNCHER_BINARY_DIR}")
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
      "-DHEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    INSTALL_COMMAND "true"
  )
elseif(USE_HEXAGON_DEVICE STREQUAL "${PICK_HW}")
  find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")
  find_hexagon_toolchain()
  file(GLOB RUNTIME_HEXAGON_DEVICE_SRCS src/runtime/hexagon/target/*.cc)

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

file(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/*.cc)
list(APPEND RUNTIME_SRCS ${RUNTIME_HEXAGON_SRCS} ${RUNTIME_HEXAGON_SIM_SRCS}
                         ${RUNTIME_HEXAGON_DEVICE_SRCS})

