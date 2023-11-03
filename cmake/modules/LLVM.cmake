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

# LLVM rules
# Due to LLVM debug symbols you can sometimes face linking issues on
# certain compiler, platform combinations if you don't set NDEBUG.
#
# See https://github.com/imageworks/OpenShadingLanguage/issues/1069
# for more discussion.
add_definitions(-DDMLC_USE_FOPEN64=0 -DNDEBUG=1)
# TODO(@jroesch, @tkonolige): if we actually use targets we can do this.
# target_compile_definitions(tvm PRIVATE NDEBUG=1)

# Test if ${USE_LLVM} is not an explicit boolean false
# It may be a boolean or a string
if(NOT ${USE_LLVM} MATCHES ${IS_FALSE_PATTERN})
  find_llvm(${USE_LLVM})
  if (${TVM_LLVM_VERSION} LESS 60)
    message(FATAL_ERROR "LLVM version 6.0 or greater is required.")
  endif()
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
  message(STATUS "Build with LLVM " ${LLVM_PACKAGE_VERSION})
  message(STATUS "Set TVM_LLVM_VERSION=" ${TVM_LLVM_VERSION})
  # Set flags that are only needed for LLVM target
  add_definitions(-DTVM_LLVM_VERSION=${TVM_LLVM_VERSION})
  if (${TVM_MLIR_VERSION})
    add_definitions(-DTVM_MLIR_VERSION=${TVM_MLIR_VERSION})
  endif()
  tvm_file_glob(GLOB COMPILER_LLVM_SRCS src/target/llvm/*.cc)
  list(APPEND TVM_LINKER_LIBS ${LLVM_LIBS})
  list(APPEND COMPILER_SRCS ${COMPILER_LLVM_SRCS})
  if(NOT MSVC)
    set_source_files_properties(${COMPILER_LLVM_SRCS}
      PROPERTIES COMPILE_DEFINITIONS "DMLC_ENABLE_RTTI=0")
    set_source_files_properties(${COMPILER_LLVM_SRCS}
      PROPERTIES COMPILE_FLAGS "-fno-rtti")
  endif()
endif()
