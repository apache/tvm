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

if(USE_XNNPACK STREQUAL "OFF")
  return()
endif()

if(IS_DIRECTORY "${USE_XNNPACK}")
  set(XNNPACK_ROOT "${USE_XNNPACK}")
  set(XNNPACK_FIND_ARGS HINTS "${XNNPACK_ROOT}" PATH_SUFFIXES include lib lib64 NO_DEFAULT_PATH)
elseif(USE_XNNPACK STREQUAL "ON")
  set(XNNPACK_FIND_ARGS)
else()
  message(FATAL_ERROR "Invalid option: USE_XNNPACK=${USE_XNNPACK}")
endif()

find_path(XNNPACK_INCLUDE_DIR xnnpack.h ${XNNPACK_FIND_ARGS})
find_library(XNNPACK_LIBRARY NAMES XNNPACK xnnpack ${XNNPACK_FIND_ARGS})
find_library(XNNPACK_MICROKERNELS_LIBRARY NAMES xnnpack-microkernels-prod
             ${XNNPACK_FIND_ARGS})
find_library(PTHREADPOOL_LIBRARY NAMES pthreadpool ${XNNPACK_FIND_ARGS})
find_library(CPUINFO_LIBRARY NAMES cpuinfo ${XNNPACK_FIND_ARGS})
find_library(KLEIDIAI_LIBRARY NAMES kleidiai ${XNNPACK_FIND_ARGS})

if(NOT XNNPACK_INCLUDE_DIR OR NOT XNNPACK_LIBRARY)
  message(FATAL_ERROR "USE_XNNPACK is enabled, but xnnpack.h or the XNNPACK library was not found")
endif()

message(STATUS "Build with XNNPACK support: ${XNNPACK_LIBRARY}")

include_directories(SYSTEM ${XNNPACK_INCLUDE_DIR})
add_definitions(-DTVM_USE_XNNPACK=1)
add_definitions(-DUSE_JSON_RUNTIME=1)

tvm_file_glob(GLOB XNNPACK_RELAX_CONTRIB_SRC src/relax/backend/contrib/xnnpack/*.cc)
list(APPEND COMPILER_SRCS ${XNNPACK_RELAX_CONTRIB_SRC})

tvm_file_glob(GLOB XNNPACK_RUNTIME_SRC src/runtime/contrib/xnnpack/*.cc)
list(APPEND RUNTIME_SRCS ${XNNPACK_RUNTIME_SRC})

list(APPEND TVM_RUNTIME_LINKER_LIBS ${XNNPACK_LIBRARY})
if(XNNPACK_MICROKERNELS_LIBRARY)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${XNNPACK_MICROKERNELS_LIBRARY})
endif()
if(PTHREADPOOL_LIBRARY)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${PTHREADPOOL_LIBRARY})
endif()
if(CPUINFO_LIBRARY)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CPUINFO_LIBRARY})
endif()
if(KLEIDIAI_LIBRARY)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${KLEIDIAI_LIBRARY})
endif()
