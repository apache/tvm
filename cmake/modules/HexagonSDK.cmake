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

set(FOUND_HEXAGON_SDK_ROOT FALSE)

macro(set_parent var)
  set(${var} ${ARGN} PARENT_SCOPE)
endmacro()

function(find_hexagon_sdk_root HEXAGON_SDK_PATH HEXAGON_ARCH)
  if(FOUND_HEXAGON_SDK_ROOT)
    return()
  endif()
  if(${ARGC} LESS "2")
    message(SEND_ERROR "Must provide Hexagon SDK path and Hexagon arch")
  endif()

  # Initial verification of the Hexagon SDK.
  message(STATUS "Checking Hexagon SDK root: ${HEXAGON_SDK_PATH}")
  file(GLOB_RECURSE VERSION_HEADERS "${HEXAGON_SDK_PATH}/*/version.h")
  if(VERSION_HEADERS)
    foreach(HEADER IN LISTS VERSION_HEADERS)
      if(HEADER MATCHES "incs/version.h$")
        set(SDK_VERSION_HEADER "${HEADER}")
        break()
      endif()
    endforeach()
    # The path is ${HEXAGON_SDK_ROOT}/incs/version.h.
    get_filename_component(TMP0 "${SDK_VERSION_HEADER}" DIRECTORY)
    get_filename_component(TMP1 "${TMP0}" DIRECTORY)
    set(HEXAGON_SDK_ROOT "${TMP1}" CACHE PATH "Root directory of Hexagon SDK")
  else()
    message(SEND_ERROR "Cannot validate Hexagon SDK in ${HEXAGON_SDK_PATH}")
  endif()

  execute_process(
    COMMAND grep "#define[ \t]*VERSION_STRING" "${SDK_VERSION_HEADER}"
    OUTPUT_VARIABLE SDK_VERSION_DEFINE)
  string(
    REGEX REPLACE ".*VERSION_STRING.* ([0-9\\.]+) .*" "\\1"
    SDK_VERSION_STRING "${SDK_VERSION_DEFINE}")

  if (SDK_VERSION_STRING MATCHES "3.5.1")
    message(SEND_ERROR "Hexagon SDK 3.5.1 is not supported")
  endif()

  # Set the Hexagon arch directory component.
  set(HEXARCH_DIR_v60 "ADSPv60MP")
  set(HEXARCH_DIR_v62 "ADSPv62MP")
  set(HEXARCH_DIR_v65 "computev65")
  set(HEXARCH_DIR_v66 "computev66")
  set(HEXARCH_DIR_v68 "computev68")
  set(HEXARCH_DIR_STR "HEXARCH_DIR_${HEXAGON_ARCH}")
  set(HEXARCH_DIR "${${HEXARCH_DIR_STR}}")

  if(NOT HEXARCH_DIR)
    message(SEND_ERROR
      "Please set HEXAGON_ARCH to one of v60, v62, v65, v66, v68")
  endif()

  # Set parent variables:
  # - HEXAGON_SDK_VERSION
  # - HEXAGON_SDK_INCLUDES
  # - HEXAGON_QURT_INCLUDES
  # - HEXAGON_RPCMEM_ROOT
  # - HEXAGON_REMOTE_ROOT
  # - HEXAGON_QAIC_EXE
  set_parent(HEXAGON_SDK_VERSION "${SDK_VERSION_STRING}")

  if(SDK_VERSION_STRING MATCHES "^3\.[0-9]+\.[0-9]+")
    # SDK 3.x.y
    if(HEXAGON_ARCH MATCHES "v6[7-9]|v[7-9][0-9]")
      message(SEND_ERROR
        "Hexagon SDK ${SDK_VERSION_STRING} does not support ${HEXAGON_ARCH}")
    endif()
    set_parent(HEXAGON_SDK_INCLUDES
      "${HEXAGON_SDK_ROOT}/incs"
      "${HEXAGON_SDK_ROOT}/incs/a1std"
      "${HEXAGON_SDK_ROOT}/incs/qlist"
      "${HEXAGON_SDK_ROOT}/incs/stddef")
    set_parent(HEXAGON_QURT_INCLUDES
      "${HEXAGON_SDK_ROOT}/libs/common/qurt/${HEXARCH_DIR}/include/posix"
      "${HEXAGON_SDK_ROOT}/libs/common/qurt/${HEXARCH_DIR}/include/qurt")
    set_parent(HEXAGON_RPCMEM_ROOT "${HEXAGON_SDK_ROOT}/libs/common/rpcmem")
    set_parent(HEXAGON_REMOTE_ROOT
      "${HEXAGON_SDK_ROOT}/libs/common/remote/ship/android_Release_aarch64")
    set_parent(HEXAGON_QAIC_EXE "${HEXAGON_SDK_ROOT}/tools/qaic/bin/qaic")
  else()
    # SDK 4.x.y.z
    if(HEXAGON_ARCH MATCHES "v6[02]")
      message(SEND_ERROR
        "Hexagon SDK ${SDK_VERSION_STRING} does not support ${HEXAGON_ARCH}")
    endif()
    set_parent(HEXAGON_SDK_INCLUDES
      "${HEXAGON_SDK_ROOT}/incs"
      "${HEXAGON_SDK_ROOT}/incs/stddef")
    set_parent(HEXAGON_QURT_INCLUDES
      "${HEXAGON_SDK_ROOT}/rtos/qurt/${HEXARCH_DIR}/include/posix"
      "${HEXAGON_SDK_ROOT}/rtos/qurt/${HEXARCH_DIR}/include/qurt")
    set_parent(HEXAGON_RPCMEM_ROOT "${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem")
    set_parent(HEXAGON_REMOTE_ROOT  # libadsprpc.so
      "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64")
    set_parent(HEXAGON_QAIC_EXE
      "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/Ubuntu16/qaic")
  endif()

  set(FOUND_HEXAGON_SDK_ROOT TRUE)
endfunction()
