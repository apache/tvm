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

# This Contribution is being provided by Qualcomm Technologies, Inc.,
# a Delaware corporation, or its subsidiary Qualcomm Innovation Center, Inc.,
# a California corporation, under certain additional terms and conditions
# pursuant to Section 5 of the Apache 2.0 license.  In this regard, with
# respect to this Contribution, the term "Work" in Section 1 of the
# Apache 2.0 license means only the specific subdirectory within the TVM repo
# (currently at https://github.com/dmlc/tvm) to which this Contribution is
# made.
# In any case, this submission is "Not a Contribution" with respect to its
# permitted use with any of the "vta" and "verilog" subdirectories in the TVM
# repo.
# Qualcomm Technologies, Inc. and Qualcomm Innovation Center, Inc. retain
# copyright of their respective Contributions.

function(find_hexagon_toolchain)
  if (NOT "${USE_HEXAGON_TOOLCHAIN}" STREQUAL "")
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
  else(HEXAGON_CLANG)
    message(SEND_ERROR "Cannot find Hexagon toolchain in ${TRY_PATH}")
  endif()
endfunction()

function(find_hexagon_sdk_root)
  message(STATUS "Checking Hexagon SDK root: ${USE_HEXAGON_SDK}")
  file(GLOB_RECURSE HEXAGON_AEESTDDEF "${USE_HEXAGON_SDK}/*/AEEStdDef.h")
  if(HEXAGON_AEESTDDEF)
    # The path is ${HEXAGON_SDK_ROOT}/incs/stddef/AEEStdDef.h.
    get_filename_component(HEXAGON_TMP0 "${HEXAGON_AEESTDDEF}" DIRECTORY)
    get_filename_component(HEXAGON_TMP1 "${HEXAGON_TMP0}" DIRECTORY)
    get_filename_component(HEXAGON_TMP2 "${HEXAGON_TMP1}" DIRECTORY)
    set(HEXAGON_SDK_ROOT "${HEXAGON_TMP2}" CACHE PATH
        "Root directory of Hexagon SDK")
  else(HEXAGON_AEESTDDEF)
    message(SEND_ERROR "Cannot validate Hexagon SDK in ${USE_HEXAGON_SDK}")
  endif()
endfunction()

if(USE_HEXAGON STREQUAL "OFF")
  return()
elseif(NOT USE_HEXAGON STREQUAL "sim" AND
       NOT USE_HEXAGON STREQUAL "device")
  message(SEND_ERROR "USE_HEXAGON must be one of [OFF|sim|device]")
  return()
endif()
# If USE_HEXAGON is set to a valid value, make sure that USE_HEXAGON_SDK
# is defined.
if (NOT USE_HEXAGON_SDK)
  message(SEND_ERROR "Please set USE_HEXAGON_SDK to the Hexagon SDK root")
  return()
endif()

if(USE_HEXAGON STREQUAL "sim")
  find_hexagon_toolchain()
  message(STATUS "Hexagon toolchain: ${HEXAGON_TOOLCHAIN}")
  add_definitions("-DHEXAGON_TOOLCHAIN=\"${HEXAGON_TOOLCHAIN}\"")
  file(GLOB RUNTIME_HEXAGON_SIM_SRCS src/runtime/hexagon/sim/*.cc)
  include_directories("${HEXAGON_TOOLCHAIN}/include/iss")
  link_directories("${HEXAGON_TOOLCHAIN}/lib/iss")
  list(APPEND TVM_RUNTIME_LINKER_LIBS "-lwrapper")
elseif(USE_HEXAGON STREQUAL "device")
  find_hexagon_sdk_root()
  find_hexagon_toolchain()
  message(STATUS "Hexagon SDK: ${HEXAGON_SDK_ROOT}")
  add_definitions("-DHEXAGON_TOOLCHAIN=\"${HEXAGON_TOOLCHAIN}\"")
  file(GLOB RUNTIME_HEXAGON_DEVICE_SRCS src/runtime/hexagon/device/*.cc)
  include_directories("${HEXAGON_SDK_ROOT}/incs/stddef")
  include_directories("${HEXAGON_SDK_ROOT}/libs/common/rpcmem/inc")
  include_directories("${HEXAGON_SDK_ROOT}/libs/common/remote/ship")
  include_directories("${HEXAGON_TOOLCHAIN}/include/iss")
  list(APPEND TVM_RUNTIME_LINKER_LIBS "-ldl")
endif()

add_definitions("-DDMLC_LOG_FATAL_THROW=0")
file(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/*.cc)
list(APPEND RUNTIME_SRCS ${RUNTIME_HEXAGON_SRCS} ${RUNTIME_HEXAGON_SIM_SRCS}
                         ${RUNTIME_HEXAGON_DEVICE_SRCS})
