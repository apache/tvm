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

if(USE_NNPACK)
  if(NNPACK_PATH STREQUAL "")
    set(NNPACK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/NNPack)
  endif()
	set(PTHREAD_POOL_PATH ${NNPACK_PATH}/deps/pthreadpool)
  tvm_file_glob(GLOB NNPACK_CONTRIB_SRC src/runtime/contrib/nnpack/*.cc)
  list(APPEND RUNTIME_SRCS ${NNPACK_CONTRIB_SRC})
	include_directories(${NNPACK_PATH}/include)
	include_directories(${PTHREAD_POOL_PATH}/include)
  find_library(NNPACK_CONTRIB_LIB nnpack ${NNPACK_PATH}/lib)
  find_library(NNPACK_PTHREAD_CONTRIB_LIB pthreadpool ${NNPACK_PATH}/lib)
  find_library(NNPACK_CPUINFO_CONTRIB_LIB cpuinfo ${NNPACK_PATH}/lib)
  find_library(NNPACK_CLOG_CONTRIB_LIB clog ${NNPACK_PATH}/lib)

  list(APPEND TVM_RUNTIME_LINKER_LIBS ${NNPACK_CONTRIB_LIB})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${NNPACK_PTHREAD_CONTRIB_LIB})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${NNPACK_CPUINFO_CONTRIB_LIB})
  if(NNPACK_CLOG_CONTRIB_LIB)
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${NNPACK_CLOG_CONTRIB_LIB})
  endif(NNPACK_CLOG_CONTRIB_LIB)
endif(USE_NNPACK)
