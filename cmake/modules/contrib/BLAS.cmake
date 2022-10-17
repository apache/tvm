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

if(USE_BLAS STREQUAL "openblas")
  find_library(BLAS_LIBRARY openblas)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS src/runtime/contrib/cblas/cblas.cc)
  message(STATUS "Using BLAS library " ${BLAS_LIBRARY})
  find_path(BLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
  if(BLAS_INCLUDE_DIR)
    message(STATUS "Using BLAS header in " ${BLAS_INCLUDE_DIR})
    include_directories(SYSTEM ${BLAS_INCLUDE_DIR})
  endif()
elseif(USE_BLAS STREQUAL "atlas" OR USE_BLAS STREQUAL "blas")
  find_library(BLAS_LIBRARY cblas)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS src/runtime/contrib/cblas/cblas.cc)
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(USE_BLAS STREQUAL "apple")
  find_library(BLAS_LIBRARY Accelerate)
  include_directories(SYSTEM ${BLAS_LIBRARY}/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS src/runtime/contrib/cblas/cblas.cc)
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(USE_BLAS STREQUAL "mkl")
  message(DEPRECATION "USE_BLAS=mkl is deprecated. Use USE_MKL=ON instead.")
  set(USE_MKL ON)
elseif(USE_BLAS STREQUAL "none")
  # pass
else()
  message(FATAL_ERROR "Invalid option: USE_BLAS=" ${USE_BLAS})
endif()

if(USE_MKL OR USE_MKL_PATH)
  if(USE_MKL_PATH)
    message(DEPRECATION "USE_MKL_PATH=${USE_MKL_PATH} is deprecated. Use USE_MKL=${USE_MKL_PATH} instead.")
  endif()
  if(NOT USE_MKL)
    set(USE_MKL ${USE_MKL_PATH})
  endif()
  if(NOT IS_DIRECTORY ${USE_MKL})
    set(USE_MKL /opt/intel/mkl)
  endif()
  if(APPLE)
    find_library(BLAS_LIBRARY_MKL NAMES mklml mkl_rt HINTS ${USE_MKL}/lib/ ${USE_MKL}/lib/intel64)
  elseif(UNIX)
    find_library(BLAS_LIBRARY_MKL NAMES mkl_rt mklml_gnu HINTS ${USE_MKL}/lib/ ${USE_MKL}/lib/intel64)
  elseif(MSVC)
    find_library(BLAS_LIBRARY_MKL NAMES mkl_rt HINTS ${USE_MKL}/lib/ ${USE_MKL}/lib/intel64_win)
  endif()
  include_directories(SYSTEM ${USE_MKL}/include)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY_MKL})
  list(APPEND RUNTIME_SRCS src/runtime/contrib/cblas/mkl.cc)
  add_definitions(-DUSE_MKL_BLAS=1)
  message(STATUS "Use MKL library " ${BLAS_LIBRARY_MKL})
endif()
