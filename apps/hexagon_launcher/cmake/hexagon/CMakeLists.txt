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
project(HexagonLauncherRPCSkel C CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/../HexagonLauncher.cmake")

add_custom_command(
  OUTPUT ${LAUNCHER_RPC_SKEL_C} ${LAUNCHER_RPC_H}
  COMMAND ${QAIC_EXE} ${QAIC_FLAGS} "${LAUNCHER_SRC}/${LAUNCHER_RPC_IDL}"
  MAIN_DEPENDENCY "${LAUNCHER_SRC}/${LAUNCHER_RPC_IDL}"
)

include_directories(SYSTEM
  ${HEXAGON_QURT_INCLUDES}
  ${CMAKE_CURRENT_BINARY_DIR}   # Output of qaic will go here
)

link_directories(${HEXAGON_QURT_LIBS})

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

add_library(launcher_rpc_skel SHARED
  "${LAUNCHER_RPC_H}"
  "${LAUNCHER_RPC_SKEL_C}"
  "${SKEL_SRCS}"
)

ExternalProject_Add(static_hexagon_tvm_runtime
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND $(MAKE) runtime
  CMAKE_ARGS
  "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
  "-DCMAKE_CXX_STANDARD=14"
  "-DUSE_LIBBACKTRACE=OFF"
  "-DUSE_LLVM=OFF"
  "-DUSE_RPC=OFF"
  "-DBUILD_STATIC_RUNTIME=ON"
  "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(static_hexagon_tvm_runtime BINARY_DIR)

add_dependencies(launcher_rpc_skel static_hexagon_tvm_runtime)
add_library(h_tvm_runtime STATIC IMPORTED)
set_target_properties(h_tvm_runtime PROPERTIES IMPORTED_LOCATION "${BINARY_DIR}/libtvm_runtime.a")

target_link_libraries(launcher_rpc_skel -Wl,--whole-archive h_tvm_runtime -Wl,--no-whole-archive)

