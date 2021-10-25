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

if(NOT DEFINED USE_HEXAGON_SDK)
  message(SEND_ERROR "Please set USE_HEXAGON_SDK to the location of Hexagon SDK")
endif()
if (NOT DEFINED USE_HEXAGON_ARCH)
  message(SEND_ERROR "Please set USE_HEXAGON_ARCH to the Hexagon architecture version")
endif()

set(TVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../")

include(ExternalProject)
include("${TVM_SOURCE_DIR}/cmake/modules/HexagonSDK.cmake")

find_hexagon_sdk_root("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}")

include_directories(SYSTEM ${HEXAGON_SDK_INCLUDES} ${HEXAGON_REMOTE_ROOT})

set(QAIC_EXE "${HEXAGON_QAIC_EXE}")
foreach(INCDIR IN LISTS HEXAGON_SDK_INCLUDES HEXAGON_REMOTE_ROOT)
  list(APPEND QAIC_FLAGS "-I${INCDIR}")
endforeach()

set(LAUNCHER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../")
set(CMAKE_SKIP_RPATH TRUE)

# Qaic for the domain header.
#
# Don't add paths to these filenames, or otherwise cmake may spontaneously
# add -o option to the qaic invocation (with an undesirable path).
set(LAUNCHER_RPC_IDL "launcher_rpc.idl")
set(LAUNCHER_RPC_H "launcher_rpc.h")
set(LAUNCHER_RPC_SKEL_C "launcher_rpc_skel.c")
set(LAUNCHER_RPC_STUB_C "launcher_rpc_stub.c")

include_directories(
  "${LAUNCHER_SRC}"
  "${TVM_SOURCE_DIR}/include"
  "${TVM_SOURCE_DIR}/3rdparty/dlpack/include"
  "${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include"
)
