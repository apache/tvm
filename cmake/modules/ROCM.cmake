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

# ROCM Module
find_rocm(${USE_ROCM})

if(ROCM_FOUND)
  # always set the includedir
  # avoid global retrigger of cmake
  include_directories(SYSTEM ${ROCM_INCLUDE_DIRS})
  add_definitions(-D__HIP_PLATFORM_HCC__=1)
endif(ROCM_FOUND)


if(USE_ROCM)
  if(NOT ROCM_FOUND)
    message(FATAL_ERROR "Cannot find ROCM, USE_ROCM=" ${USE_ROCM})
  endif()
  message(STATUS "Build with ROCM support")
  tvm_file_glob(GLOB RUNTIME_ROCM_SRCS src/runtime/rocm/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_ROCM_SRCS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${ROCM_HIPHCC_LIBRARY})
  if (ROCM_HSA_LIBRARY)
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${ROCM_HSA_LIBRARY})
  endif()

  if(USE_MIOPEN)
    message(STATUS "Build with MIOpen support")
    tvm_file_glob(GLOB MIOPEN_CONTRIB_SRCS src/runtime/contrib/miopen/*.cc)
    list(APPEND RUNTIME_SRCS ${MIOPEN_CONTRIB_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${ROCM_MIOPEN_LIBRARY})
  endif(USE_MIOPEN)

  if(USE_ROCBLAS)
    message(STATUS "Build with RocBLAS support")
    tvm_file_glob(GLOB ROCBLAS_CONTRIB_SRCS src/runtime/contrib/rocblas/*.cc)
    list(APPEND RUNTIME_SRCS ${ROCBLAS_CONTRIB_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${ROCM_ROCBLAS_LIBRARY})
  endif(USE_ROCBLAS)

  if(USE_THRUST)
    message(STATUS "Build with rocThrust support")
    # We need to override CXX to hipcc. This is required by rocthrust
    if (${CMAKE_CXX_COMPILER} MATCHES "hipcc$")
      message(STATUS "Using hipcc compiler to compile rocthrust code.")
    else()
      message(FATAL_ERROR "Set CXX=hipcc to compile rocthrust code.")
    endif()

    find_package(rocprim REQUIRED)
    find_package(rocthrust REQUIRED)
    set_source_files_properties(src/runtime/contrib/thrust/thrust.cu PROPERTIES LANGUAGE CXX)
    list(APPEND RUNTIME_SRCS src/runtime/contrib/thrust/thrust.cu)
    list(APPEND TVM_RUNTIME_LINKER_LIBS roc::rocthrust)
  endif(USE_THRUST)

else(USE_ROCM)
  list(APPEND COMPILER_SRCS src/target/opt/build_rocm_off.cc)
endif(USE_ROCM)
