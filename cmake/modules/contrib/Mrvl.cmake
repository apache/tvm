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

if(USE_MRVL)
  IF (MRVL_COMPILER_LIB_PATH)
    add_definitions(-DTVM_USE_MRVL_COMPILER_LIB=1)
    # copy 3 pre-built static lib files of Marvell compiler-backend
    #   under the MRVL_COMPILER_LIB_PATH directory
    file(COPY ${MRVL_COMPILER_LIB_PATH}/libmrvlcompiler.a
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${MRVL_COMPILER_LIB_PATH}/libml.a
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${MRVL_COMPILER_LIB_PATH}/libnum.a
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(COPY ${MRVL_COMPILER_LIB_PATH}/libisa.a
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(GLOB MRVL_RUNTIME_LIB
      ${CMAKE_CURRENT_BINARY_DIR}/libmrvlcompiler.a
      ${CMAKE_CURRENT_BINARY_DIR}/libml.a
      ${CMAKE_CURRENT_BINARY_DIR}/libisa.a
      ${CMAKE_CURRENT_BINARY_DIR}/libnum.a
    )
    # FIXME: list(APPEND TVM_LINKER_LIBS ${MRVL_LIB})
    message(STATUS "Build with 4 Mrvl lib *.a files: ${MRVL_RUNTIME_LIB}")
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${MRVL_RUNTIME_LIB})
  ENDIF (MRVL_COMPILER_LIB_PATH) 

  # Mrvl Module
  message(STATUS "Build with Mrvl support")
  add_definitions(-DTVM_USE_MRVL=1)
  # FIXME: find_livrary(MRVL_LIB Mrvl)
  # FIXME: find_livrary(MRVL_RUNTIME_LIB Mrvl_runtime)
  file(GLOB RUNTIME_MRVL_SRCS
    src/runtime/contrib/mrvl/mrvl_runtime.cc
  )
  list(APPEND RUNTIME_SRCS ${RUNTIME_MRVL_SRCS})

  file(GLOB COMPILER_MRVL_SRCS
    src/relay/backend/contrib/mrvl/graph_executor_codegen_mrvl.cc
    src/relay/backend/contrib/mrvl/codegen.cc
    src/relay/backend/contrib/mrvl/drop_noop_transpose.cc
  )
  list(APPEND COMPILER_SRCS ${COMPILER_MRVL_SRCS})

endif(USE_MRVL)
