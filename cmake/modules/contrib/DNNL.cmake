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

if(IS_DIRECTORY ${USE_DNNL})
  find_library(EXTERN_LIBRARY_DNNL NAMES dnnl HINTS ${USE_DNNL}/lib/)
  if (EXTERN_LIBRARY_DNNL STREQUAL "EXTERN_LIBRARY_DNNL-NOTFOUND")
    message(WARNING "Cannot find DNNL library at ${USE_DNNL}.")
  else()
    add_definitions(-DUSE_JSON_RUNTIME=1)
    tvm_file_glob(GLOB DNNL_RELAY_CONTRIB_SRC src/relay/backend/contrib/dnnl/*.cc)
    list(APPEND COMPILER_SRCS ${DNNL_RELAY_CONTRIB_SRC})

    list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_DNNL})
    tvm_file_glob(GLOB DNNL_CONTRIB_SRC src/runtime/contrib/dnnl/dnnl_json_runtime.cc
                                        src/runtime/contrib/dnnl/dnnl_utils.cc
                                        src/runtime/contrib/dnnl/dnnl.cc
                                        src/runtime/contrib/cblas/dnnl_blas.cc)
    list(APPEND RUNTIME_SRCS ${DNNL_CONTRIB_SRC})
    message(STATUS "Build with DNNL JSON runtime: " ${EXTERN_LIBRARY_DNNL})
  endif()
elseif((USE_DNNL STREQUAL "ON") OR (USE_DNNL STREQUAL "JSON"))
  add_definitions(-DUSE_JSON_RUNTIME=1)
  tvm_file_glob(GLOB DNNL_RELAY_CONTRIB_SRC src/relay/backend/contrib/dnnl/*.cc)
  list(APPEND COMPILER_SRCS ${DNNL_RELAY_CONTRIB_SRC})

  find_library(EXTERN_LIBRARY_DNNL dnnl)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_DNNL})
  tvm_file_glob(GLOB DNNL_CONTRIB_SRC src/runtime/contrib/dnnl/dnnl_json_runtime.cc
                                      src/runtime/contrib/dnnl/dnnl_utils.cc
                                      src/runtime/contrib/dnnl/dnnl.cc
                                      src/runtime/contrib/cblas/dnnl_blas.cc)
  list(APPEND RUNTIME_SRCS ${DNNL_CONTRIB_SRC})
  message(STATUS "Build with DNNL JSON runtime: " ${EXTERN_LIBRARY_DNNL})
elseif(USE_DNNL STREQUAL "C_SRC")
  tvm_file_glob(GLOB DNNL_RELAY_CONTRIB_SRC src/relay/backend/contrib/dnnl/*.cc)
  list(APPEND COMPILER_SRCS ${DNNL_RELAY_CONTRIB_SRC})

  find_library(EXTERN_LIBRARY_DNNL dnnl)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_DNNL})
  tvm_file_glob(GLOB DNNL_CONTRIB_SRC src/runtime/contrib/dnnl/dnnl.cc
                                      src/runtime/contrib/cblas/dnnl_blas.cc)
  list(APPEND RUNTIME_SRCS ${DNNL_CONTRIB_SRC})
  message(STATUS "Build with DNNL C source module: " ${EXTERN_LIBRARY_DNNL})
elseif(USE_DNNL STREQUAL "OFF")
  # pass
else()
  message(FATAL_ERROR "Invalid option: USE_DNNL=" ${USE_DNNL})
endif()
