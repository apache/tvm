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
project(HexagonAndroidRPC C CXX)

include("${CMAKE_CURRENT_SOURCE_DIR}/../HexagonRPC.cmake")

add_custom_command(
  OUTPUT ${HEXAGON_PROXY_RPC_STUB_C} ${HEXAGON_PROXY_RPC_H}
  COMMAND ${QAIC_EXE} ${QAIC_FLAGS} "${HEXAGON_PROXY_RPC_SRC}/${HEXAGON_PROXY_RPC_IDL}"
  MAIN_DEPENDENCY "${HEXAGON_PROXY_RPC_SRC}/${HEXAGON_PROXY_RPC_IDL}"
)

include_directories(SYSTEM
  "${HEXAGON_SDK_INCLUDES}"
  "${HEXAGON_RPCMEM_ROOT}/inc"
  "${CMAKE_CURRENT_BINARY_DIR}"   # Output of qaic will go here
)

link_directories(${HEXAGON_REMOTE_ROOT})

add_definitions(-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

set(TVM_RPC_ENV_SOURCES
  ${HEXAGON_PROXY_RPC_SRC}/rpc_env.cc
)

add_library(rpc_env SHARED
  ${TVM_RPC_ENV_SOURCES}
  ${HEXAGON_PROXY_RPC_H}
  ${HEXAGON_PROXY_RPC_STUB_C}
)

ExternalProject_Add(android_tvm_runtime
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND $(MAKE) runtime
  CMAKE_ARGS
  "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
  "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
  "-DANDROID_ABI=${ANDROID_ABI}"
  "-DCMAKE_CXX_STANDARD=14"
  "-DUSE_LIBBACKTRACE=OFF"
  "-DUSE_LLVM=OFF"
  "-DUSE_RPC=ON"
  "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
  "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(android_tvm_runtime BINARY_DIR)
ExternalProject_Add_Step(android_tvm_runtime copy_binaries
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/libtvm_runtime.so
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDEES install
)

add_dependencies(rpc_env android_tvm_runtime)
add_library(a_tvm_runtime SHARED IMPORTED)
set_target_properties(a_tvm_runtime PROPERTIES IMPORTED_LOCATION "${BINARY_DIR}/libtvm_runtime.so")

target_link_libraries(rpc_env cdsprpc log a_tvm_runtime)

# TVM CPP RPC build
set(TVM_RPC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp_rpc")


set(TVM_RPC_SOURCES
  ${TVM_RPC_DIR}/main.cc
  ${TVM_RPC_DIR}/rpc_server.cc
)

# Set output to same directory as the other TVM libs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_executable(tvm_rpc ${TVM_RPC_SOURCES})


target_include_directories(
  tvm_rpc
  PUBLIC "${TVM_RPC_DIR}../../include"
  PUBLIC "${TVM_RPC_DIR}../../3rdparty/dlpack"
  PUBLIC "${TVM_RPC_DIR}../../3rdparty/dmlc-core"
)

add_dependencies(rpc_env android_tvm_runtime)
target_link_libraries(rpc_env a_tvm_runtime)

add_dependencies(tvm_rpc android_tvm_runtime rpc_env)
target_link_libraries(tvm_rpc a_tvm_runtime rpc_env)
