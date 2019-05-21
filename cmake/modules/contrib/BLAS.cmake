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

# Plugin rules for cblas
file(GLOB CBLAS_CONTRIB_SRC src/contrib/cblas/*.cc)

if(USE_BLAS STREQUAL "openblas")
  find_library(BLAS_LIBRARY openblas)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS ${CBLAS_CONTRIB_SRC})
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(USE_BLAS STREQUAL "mkl")
  if(NOT IS_DIRECTORY ${USE_MKL_PATH})
    set(USE_MKL_PATH /opt/intel/mkl)
  endif()
  if(APPLE)
    find_library(BLAS_LIBRARY NAMES mklml HINTS ${USE_MKL_PATH}/lib/ ${USE_MKL_PATH}/lib/intel64)
  elseif(UNIX)
    find_library(BLAS_LIBRARY NAMES mkl_rt mklml_gnu HINTS ${USE_MKL_PATH}/lib/ ${USE_MKL_PATH}/lib/intel64)
  endif()
  include_directories(${USE_MKL_PATH}/include)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS ${CBLAS_CONTRIB_SRC})
  add_definitions(-DUSE_MKL_BLAS=1)
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(USE_BLAS STREQUAL "atlas" OR USE_BLAS STREQUAL "blas")
  find_library(BLAS_LIBRARY cblas)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS ${CBLAS_CONTRIB_SRC})
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(USE_BLAS STREQUAL "apple")
  find_library(BLAS_LIBRARY Accelerate)
  include_directories(${BLAS_LIBRARY}/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS ${CBLAS_CONTRIB_SRC})
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(USE_BLAS STREQUAL "none")
  # pass
else()
  message(FATAL_ERROR "Invalid option: USE_BLAS=" ${USE_BLAS})
endif()
