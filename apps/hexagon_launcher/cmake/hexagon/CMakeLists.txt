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
project(HexagonLauncherRPCSkel C CXX ASM)

include("${CMAKE_CURRENT_SOURCE_DIR}/../HexagonLauncher.cmake")
# From the include above get
#   SDK_INCLUDE_DIRS
#   QAIC_EXE_PATH
# and
#   QAIC_FLAGS
#   LAUNCHER_SRC
#   LAUNCHER_RPC_IDL
#   LAUNCHER_RPC_H
#   LAUNCHER_RPC_SKEL_C
#   LAUNCHER_RPC_STUB_C

add_custom_command(
  OUTPUT ${LAUNCHER_RPC_SKEL_C} ${LAUNCHER_RPC_H}
  COMMAND ${QAIC_EXE_PATH} ${QAIC_FLAGS} "${LAUNCHER_SRC}/${LAUNCHER_RPC_IDL}"
  MAIN_DEPENDENCY "${LAUNCHER_SRC}/${LAUNCHER_RPC_IDL}"
)

get_hexagon_sdk_property("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}"
  QURT_INCLUDE QURT_INCLUDE_DIRS
  QURT_LIB     QURT_LIB_DIRS
)
if(NOT QURT_INCLUDE_DIRS OR NOT QURT_LIB_DIRS)
  message(WARNING "Could not locate some Hexagon SDK components")
endif()

include_directories(SYSTEM
  ${QURT_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}   # Output of qaic will go here
)

link_directories(${QURT_LIB_DIRS})

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

set(SKEL_SRCS
  "${LAUNCHER_SRC}/launcher_core.cc"
  "${LAUNCHER_SRC}/launcher_hexagon.cc"
)
set(PROFILER_DIR "${TVM_SOURCE_DIR}/src/runtime/hexagon/profiler")

add_library(launcher_rpc_skel SHARED
  "${LAUNCHER_RPC_H}"
  "${LAUNCHER_RPC_SKEL_C}"
  "${SKEL_SRCS}"
  "${PROFILER_DIR}/prof_utils.cc"
  "${PROFILER_DIR}/lwp_handler.S"
)

ExternalProject_Add(static_hexagon_tvm_runtime
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND $(MAKE) runtime
  CMAKE_ARGS
  "-DBUILD_STATIC_RUNTIME=ON"
  "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_CXX_STANDARD=17"
  "-DUSE_HEXAGON=ON"
  "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
  "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
  "-DUSE_LIBBACKTRACE=OFF"
  "-DUSE_LLVM=OFF"
  "-DUSE_RPC=OFF"
  "-DUSE_CUSTOM_LOGGING=ON"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(static_hexagon_tvm_runtime BINARY_DIR)

add_dependencies(launcher_rpc_skel static_hexagon_tvm_runtime)
add_library(h_tvm_runtime STATIC IMPORTED)
set_target_properties(h_tvm_runtime PROPERTIES IMPORTED_LOCATION "${BINARY_DIR}/libtvm_runtime.a")

target_link_libraries(launcher_rpc_skel -Wl,--whole-archive h_tvm_runtime -Wl,--no-whole-archive)
