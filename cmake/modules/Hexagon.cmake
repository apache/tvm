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
  if(USE_HEXAGON_LAUNCHER STREQUAL "OFF" AND
      USE_HEXAGON_PROXY_RPC STREQUAL "OFF" AND NOT USE_HEXAGON_RPC)
    if(USE_HEXAGON_DEVICE STREQUAL "OFF")
      list(APPEND COMPILER_SRCS src/target/opt/build_hexagon_off.cc)
      # append select runtime sources for unit testing
      list(APPEND RUNTIME_SRCS src/runtime/hexagon/hexagon/hexagon_buffer.cc)
      list(APPEND RUNTIME_SRCS src/runtime/hexagon/hexagon/hexagon_common.cc)
      if (NOT USE_HEXAGON_RPC)
        return()
      endif()
    elseif(NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_SIM}" AND
        NOT USE_HEXAGON_DEVICE STREQUAL "${PICK_HW}")
      set(ERROR_MSG
        "USE_HEXAGON_DEVICE must be one of [${PICK_NONE}|${PICK_SIM}|${PICK_HW}]")
      message(SEND_ERROR "${ERROR_MSG}")
      return()
    endif()
  endif()
endif()

# If USE_HEXAGON_DEVICE/LAUNCHER is set to a valid value, make sure that USE_HEXAGON_SDK
# is defined.
if(NOT USE_HEXAGON_SDK)
  message(SEND_ERROR "Please set USE_HEXAGON_SDK to the Hexagon SDK root")
  return()
endif()

if(USE_HEXAGON_LAUNCHER STREQUAL "ON" OR
    USE_HEXAGON_PROXY_RPC STREQUAL "ON")
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
endif()

if(USE_HEXAGON_LAUNCHER STREQUAL "ON")
  set(LAUNCHER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/apps_hexagon_launcher")
  ExternalProject_Add(launcher_android
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/apps/hexagon_launcher/cmake/android"
    INSTALL_DIR "${LAUNCHER_BINARY_DIR}"
    BUILD_ALWAYS ON
    CMAKE_ARGS
    "-DCMAKE_TOOLCHAIN_FILE=${USE_ANDROID_TOOLCHAIN}"
    "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
    "-DANDROID_ABI=${ANDROID_ABI}"
    "-DFASTRPC_LIBS=STUB"
    "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(launcher_android BINARY_DIR)
  ExternalProject_Add_Step(launcher_android copy_binaries
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${BINARY_DIR}/launcher_android ${BINARY_DIR}/libtvm_runtime.so
      ${LAUNCHER_BINARY_DIR}
    DEPENDEES install
  )
  ExternalProject_Add(launcher_hexagon
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/apps/hexagon_launcher/cmake/hexagon"
    INSTALL_DIR "${LAUNCHER_BINARY_DIR}"
    BUILD_ALWAYS ON
    CMAKE_ARGS
    "-DCMAKE_C_COMPILER=${USE_HEXAGON_TOOLCHAIN}/bin/hexagon-clang"
    "-DCMAKE_CXX_COMPILER=${USE_HEXAGON_TOOLCHAIN}/bin/hexagon-clang++"
    "-DFASTRPC_LIBS=SKEL"
    "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(launcher_hexagon BINARY_DIR)
  ExternalProject_Add_Step(launcher_hexagon copy_binaries
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${BINARY_DIR}/liblauncher_rpc_skel.so
      ${LAUNCHER_BINARY_DIR}
    DEPENDEES install
  )

  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${LAUNCHER_BINARY_DIR}")

endif()

if(USE_HEXAGON_PROXY_RPC STREQUAL "ON")
  set(RPC_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/apps_hexagon_proxy_rpc")
  ExternalProject_Add(proxy_rpc_android
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/apps/hexagon_proxy_rpc/cmake/android"
    INSTALL_DIR "${RPC_BINARY_DIR}"
    BUILD_ALWAYS ON
    CMAKE_ARGS
    "-DCMAKE_TOOLCHAIN_FILE=${USE_ANDROID_TOOLCHAIN}"
    "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
    "-DANDROID_ABI=${ANDROID_ABI}"
    "-DFASTRPC_LIBS=STUB"
    "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(proxy_rpc_android BINARY_DIR)
  ExternalProject_Add_Step(proxy_rpc_android copy_binaries
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${BINARY_DIR}/libtvm_runtime.so ${BINARY_DIR}/librpc_env.so ${BINARY_DIR}/tvm_rpc
      ${RPC_BINARY_DIR}
    DEPENDEES install
  )
  ExternalProject_Add(proxy_rpc_hexagon
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/apps/hexagon_proxy_rpc/cmake/hexagon"
    INSTALL_DIR "${RPC_BINARY_DIR}"
    BUILD_ALWAYS ON
    CMAKE_ARGS
    "-DCMAKE_C_COMPILER=${USE_HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang"
    "-DCMAKE_CXX_COMPILER=${USE_HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang++"
    "-DFASTRPC_LIBS=SKEL"
    "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    INSTALL_COMMAND ""
  )
  ExternalProject_Get_Property(proxy_rpc_hexagon BINARY_DIR)
  ExternalProject_Add_Step(proxy_rpc_hexagon copy_binaries
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${BINARY_DIR}/libhexagon_proxy_rpc_skel.so
      ${RPC_BINARY_DIR}
    DEPENDEES install
  )

  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${RPC_BINARY_DIR}")
endif()

if(USE_HEXAGON_RPC)
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
      "RPC server for Hexagon.")
  endif()

  if(NOT DEFINED USE_HEXAGON_SDK)
    message(SEND_ERROR "Please set USE_HEXAGON_SDK to build the android "
      "RPC server for Hexagon RPC.")
  endif()
  if(NOT DEFINED USE_HEXAGON_ARCH)
    message(SEND_ERROR "Please set USE_HEXAGON_ARCH to build the android "
      "RPC server for Hexagon RPC.")
  endif()
  find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")

  set(HEXAGON_RPC_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/hexagon_rpc")
  file(MAKE_DIRECTORY ${HEXAGON_RPC_OUTPUT})

  # Android Part
  ExternalProject_Add(android_runtime_rpc
    SOURCE_DIR "${CMAKE_SOURCE_DIR}"
    BUILD_COMMAND $(MAKE) runtime tvm_rpc
    CMAKE_ARGS
    "-DCMAKE_TOOLCHAIN_FILE=${USE_ANDROID_TOOLCHAIN}"
    "-DUSE_ANDROID_TOOLCHAIN=${USE_ANDROID_TOOLCHAIN}"
    "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
    "-DANDROID_ABI=${ANDROID_ABI}"
    "-DCMAKE_CXX_STANDARD=14"
    "-DUSE_LIBBACKTRACE=OFF"
    "-DUSE_LLVM=OFF"
    "-DUSE_RPC=ON"
    "-DUSE_CPP_RPC=ON"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    "-DCMAKE_VERBOSE_MAKEFILE=ON"
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
  )
  ExternalProject_Get_Property(android_runtime_rpc BINARY_DIR)
  ExternalProject_Add_Step(android_runtime_rpc copy_binary_runtime
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${BINARY_DIR}/libtvm_runtime.so
      ${HEXAGON_RPC_OUTPUT}/libtvm_runtime.so
    DEPENDEES install
  )
  ExternalProject_Add_Step(android_runtime_rpc copy_binary_rpc
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${BINARY_DIR}/tvm_rpc
      ${HEXAGON_RPC_OUTPUT}/tvm_rpc_android
    DEPENDEES install
  )

  if("${USE_HEXAGON_TOOLCHAIN}" STREQUAL "")
    message(SEND_ERROR "Please set USE_HEXAGON_TOOLCHAIN to build the hexagon "
      "RPC SKEL.")
  endif()
  find_hexagon_toolchain()
  message(STATUS "HEXAGON_TOOLCHAIN: ${HEXAGON_TOOLCHAIN}")

  # Hexagon Part
  ExternalProject_Add(hexagon_rpc_skel
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/cmake/libs/hexagon_rpc_skel"
    INSTALL_DIR "${LAUNCHER_BINARY_DIR}"
    CMAKE_ARGS
    "-DCMAKE_C_COMPILER=${HEXAGON_TOOLCHAIN}/bin/hexagon-clang"
    "-DCMAKE_CXX_COMPILER=${HEXAGON_TOOLCHAIN}/bin/hexagon-clang++"
    "-DFASTRPC_LIBS=SKEL"
    "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
  )
  ExternalProject_Get_Property(hexagon_rpc_skel BINARY_DIR)
  ExternalProject_Add_Step(hexagon_rpc_skel copy_hexagon_skel
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${BINARY_DIR}/libhexagon_rpc_skel.so
      ${HEXAGON_RPC_OUTPUT}/libhexagon_rpc_skel.so
    DEPENDEES install
  )

  # copy android_bash template file
  configure_file("${CMAKE_SOURCE_DIR}/src/runtime/hexagon/rpc/android_bash.sh.template" 
    ${HEXAGON_RPC_OUTPUT} COPYONLY)

  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${HEXAGON_RPC_OUTPUT}")

  # Used in `src/target/llvm/llvm_common.h`
  add_definitions(-DTVM_USE_HEXAGON_LLVM)
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

set(RUNTIME_HEXAGON_COMMON_SRCS src/runtime/hexagon/hexagon_module.cc)
if (USE_HEXAGON_DEVICE STREQUAL "${PICK_NONE}")
  if(BUILD_FOR_HEXAGON)
    file(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/hexagon/*.cc)
  elseif(BUILD_FOR_ANDROID AND HEXAGON_SDK_PATH_DEFINED)
    list(APPEND RUNTIME_HEXAGON_SRCS src/runtime/hexagon/proxy_rpc/device_api.cc)
  else()
  file(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/host/*.cc)
  endif()
else()
  file(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/android/*.cc)
endif()

if(USE_HEXAGON_RPC)
  file(GLOB RUNTIME_HEXAGON_SRCS src/runtime/hexagon/host/*.cc)
endif()

if(USE_HEXAGON_SDK AND BUILD_FOR_ANDROID)
  find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")
  include_directories(SYSTEM ${HEXAGON_SDK_INCLUDES} ${HEXAGON_REMOTE_ROOT})

  set(QAIC_EXE "${HEXAGON_QAIC_EXE}")
  foreach(INCDIR IN LISTS HEXAGON_SDK_INCLUDES HEXAGON_REMOTE_ROOT)
    list(APPEND QAIC_FLAGS "-I${INCDIR}")
  endforeach()

  set(HEXAGON_RPC_DIR "${CMAKE_SOURCE_DIR}/src/runtime/hexagon/rpc")
  set(RPC_IDL "hexagon_rpc.idl")
  set(RPC_H "hexagon_rpc.h")
  set(RPC_STUB_C "hexagon_rpc_stub.c")
  
  add_custom_command(
    OUTPUT "${HEXAGON_RPC_DIR}/${RPC_STUB_C}" "${HEXAGON_RPC_DIR}/${RPC_H}"
    COMMAND ${QAIC_EXE} ${QAIC_FLAGS} "${HEXAGON_RPC_DIR}/${RPC_IDL}" -o ${HEXAGON_RPC_DIR}
    MAIN_DEPENDENCY "${HEXAGON_RPC_DIR}/${RPC_IDL}"
  )
  file(GLOB HEXAGON_RPC_CPP "${HEXAGON_RPC_DIR}/android/*.cc")
  set(HEXAGON_RPC_STUB_C "${HEXAGON_RPC_DIR}/${RPC_STUB_C}")
endif()

list(APPEND RUNTIME_SRCS ${RUNTIME_HEXAGON_SRCS} ${RUNTIME_HEXAGON_SIM_SRCS}
                         ${RUNTIME_HEXAGON_DEVICE_SRCS} ${HEXAGON_RPC_CPP} ${HEXAGON_RPC_STUB_C} 
                         ${RUNTIME_HEXAGON_COMMON_SRCS})
