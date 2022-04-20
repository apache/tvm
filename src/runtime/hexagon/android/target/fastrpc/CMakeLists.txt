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
project(HexagonIDL C CXX)

if(NOT "${FASTRPC_LIBS}" STREQUAL "SKEL" AND
   NOT "${FASTRPC_LIBS}" STREQUAL "STUB")
  message(SEND_ERROR "Please set FASTRPC_LIBS to either SKEL or STUB")
endif()

include(../../../../../../cmake/utils/Utils.cmake)
include(../../../../../../cmake/modules/HexagonSDK.cmake)

get_hexagon_sdk_property("${HEXAGON_SDK_ROOT}" "${HEXAGON_ARCH}"
  SDK_INCLUDE  SDK_INCLUDE_DIRS
  QURT_INCLUDE QURT_INCLUDE_DIRS
  DSPRPC_LIB   DSPRPC_LIB_DIRS
  RPCMEM_ROOT  RPCMEM_ROOT_DIR
  QAIC_EXE     QAIC_EXE_PATH
)
if(NOT SDK_INCLUDE_DIRS OR NOT QURT_INCLUDE_DIRS OR NOT DSPRPC_LIB_DIRS OR
   NOT RPCMEM_ROOT_DIR OR NOT QAIC_EXE_PATH)
  message(WARNING "Could not locate some Hexagon SDK components")
endif()

include_directories(include)
include_directories(SYSTEM ${SDK_INCLUDE_DIRS})

foreach(INCDIR IN LISTS SDK_INCLUDE_DIRS)
  list(APPEND QAIC_FLAGS "-I${INCDIR}")
endforeach()

set(FASTRPC_SRC "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_SKIP_RPATH TRUE)

# Qaic for the non-domain header.
#
# Don't add paths to these filenames, or otherwise cmake may spontaneously
# add -o option to the qaic invocation (with an undesirable path).
set(TVM_REMOTE_ND_IDL "tvm_remote_nd.idl")
set(TVM_REMOTE_ND_H "tvm_remote_nd.h")
set(TVM_REMOTE_ND_SKEL_C "tvm_remote_nd_skel.c")
set(TVM_REMOTE_ND_STUB_C "tvm_remote_nd_stub.c")

add_custom_command(
  OUTPUT ${TVM_REMOTE_ND_SKEL_C} ${TVM_REMOTE_ND_STUB_C}
         "${FASTRPC_SRC}/include/${TVM_REMOTE_ND_H}"
  COMMAND ${QAIC_EXE_PATH} ${QAIC_FLAGS}
          "${FASTRPC_SRC}/include/${TVM_REMOTE_ND_IDL}"
  COMMAND ${CMAKE_COMMAND} -E rename "${TVM_REMOTE_ND_H}"
          "${FASTRPC_SRC}/include/${TVM_REMOTE_ND_H}"
  MAIN_DEPENDENCY "${FASTRPC_SRC}/include/${TVM_REMOTE_ND_IDL}"
)

# Qaic for the domain header.
#
# Don't add paths to these filenames, or otherwise cmake may spontaneously
# add -o option to the qaic invocation (with an undesirable path).
set(TVM_REMOTE_D_IDL "tvm_remote.idl")
set(TVM_REMOTE_D_H "tvm_remote.h")
set(TVM_REMOTE_D_SKEL_C "tvm_remote_skel.c")
set(TVM_REMOTE_D_STUB_C "tvm_remote_stub.c")

add_custom_command(
  OUTPUT ${TVM_REMOTE_D_SKEL_C} ${TVM_REMOTE_D_STUB_C}
         "${FASTRPC_SRC}/include/${TVM_REMOTE_D_H}"
  COMMAND ${QAIC_EXE_PATH} ${QAIC_FLAGS}
          "${FASTRPC_SRC}/include/${TVM_REMOTE_D_IDL}"
  COMMAND ${CMAKE_COMMAND} -E rename "${TVM_REMOTE_D_H}"
          "${FASTRPC_SRC}/include/${TVM_REMOTE_D_H}"
  MAIN_DEPENDENCY "${FASTRPC_SRC}/include/${TVM_REMOTE_D_IDL}"
)


if("${FASTRPC_LIBS}" STREQUAL "SKEL")
  # Skel libraries.
  #
  include_directories(SYSTEM ${QURT_INCLUDE_DIRS})

  # Extra compile flags (both C and C++).
  set(EXTRA_COMP_FLAGS
    "-O3"
    "-m${HEXAGON_ARCH}"
  )
  string(REGEX REPLACE ";" " " EXTRA_COMP_FLAGS_STR "${EXTRA_COMP_FLAGS}")
  set(CMAKE_C_FLAGS "${EXTRA_COMP_FLAGS_STR} ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${EXTRA_COMP_FLAGS_STR} ${CMAKE_CXX_FLAGS}")

  set(EXTRA_LINK_FLAGS
    "-Wl,--no-threads"
    "-Wl,--wrap=malloc"
    "-Wl,--wrap=calloc"
    "-Wl,--wrap=free"
    "-Wl,--wrap=realloc"
    "-Wl,--wrap=memalign"
    "-Wl,--wrap=posix_memalign"
    "-Wl,--wrap=__stack_chk_fail"
  )
  string(REGEX REPLACE ";" " " EXTRA_LINK_FLAGS_STR "${EXTRA_LINK_FLAGS}")

  set(SKEL_ND_SRCS
    "src/tvm_hvx.cc"
    "src/tvm_remote_nd_imp.cc"
  )
  add_library(tvm_remote_nd_skel SHARED
    "${FASTRPC_SRC}/include/${TVM_REMOTE_ND_H}"
    "${TVM_REMOTE_ND_SKEL_C}"
    "${SKEL_ND_SRCS}"
  )

  set(SKEL_D_SRCS
    # Also includes src/tvm_remote_nd_imp.cc
    "${SKEL_ND_SRCS}"
    "src/tvm_remote_imp.cc"
  )
  add_library(tvm_remote_skel SHARED
    "${FASTRPC_SRC}/include/${TVM_REMOTE_D_H}"
    "${TVM_REMOTE_D_SKEL_C}"
    "${SKEL_D_SRCS}"
  )

  # Separate shared library with __wrap_pthread_create.
  # It is necessary to have it as a separate library because it defines
  # a function that libtvm_runtime.so will call. Because of that, this
  # function needs to be in the global dynamic symbol table, but the
  # skel libraries are loaded as private by FastRPC.
  set(WRAP_PTHREAD_SRCS "src/tvm_wrap_pthread.cc")
  add_library(tvm_wrap_pthread SHARED ${WRAP_PTHREAD_SRCS})

  # Extra linker flags for linking shared libraries.
  set_target_properties(tvm_remote_nd_skel PROPERTIES LINK_FLAGS ${EXTRA_LINK_FLAGS_STR})
  set_target_properties(tvm_remote_skel PROPERTIES LINK_FLAGS ${EXTRA_LINK_FLAGS_STR})
  set_target_properties(tvm_wrap_pthread PROPERTIES LINK_FLAGS ${EXTRA_LINK_FLAGS_STR})
else()
  # Stub libraries.
  #
  include_directories(SYSTEM
    ${SDK_INCLUDE_DIRS}
    "${RPCMEM_ROOT_DIR}/inc"
  )
  link_directories(${DSPRPC_LIB_DIRS})

  if(RPCMEM_ROOT_DIR)
    set(RPCMEM_ANDROID_C "${RPCMEM_ROOT_DIR}/src/rpcmem_android.c")
  endif()
  add_library(tvm_remote_nd_stub SHARED
    "${FASTRPC_SRC}/include/${TVM_REMOTE_ND_H}"
    "${RPCMEM_ANDROID_C}"
    "${TVM_REMOTE_ND_STUB_C}"
  )
  add_library(tvm_remote_stub SHARED
    "${FASTRPC_SRC}/include/${TVM_REMOTE_D_H}"
    "${RPCMEM_ANDROID_C}"
    "${TVM_REMOTE_D_STUB_C}"
  )
  target_link_libraries(tvm_remote_nd_stub adsprpc)
  target_link_libraries(tvm_remote_stub adsprpc)
endif()
