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
  # avoid global retrigger of CMake
  include_directories(SYSTEM ${ROCM_INCLUDE_DIRS})
  add_definitions(-D__HIP_PLATFORM_HCC__=1)
  add_definitions(-D__HIP_PLATFORM_AMD__=1)
endif(ROCM_FOUND)

if(USE_ROCM)
  if(NOT ROCM_FOUND)
    message(FATAL_ERROR "Cannot find ROCM, USE_ROCM=" ${USE_ROCM})
  endif()
  message(STATUS "Build rocm device runtime")

  tvm_file_glob(GLOB RUNTIME_ROCM_SRCS src/runtime/rocm/*.cc)

  set(_rocm_libs ${ROCM_HIPHCC_LIBRARY})
  if(ROCM_HSA_LIBRARY)
    list(APPEND _rocm_libs ${ROCM_HSA_LIBRARY})
  endif()

  add_library(tvm_runtime_rocm_objs OBJECT ${RUNTIME_ROCM_SRCS})
  target_link_libraries(tvm_runtime_rocm_objs PUBLIC tvm_ffi_header)
  set_target_properties(tvm_runtime_rocm_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(TVM_VISIBILITY_FLAG)
    target_compile_options(tvm_runtime_rocm_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
  endif()
  add_library(tvm_runtime_rocm SHARED $<TARGET_OBJECTS:tvm_runtime_rocm_objs>)
  list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_rocm)
  target_link_libraries(tvm_runtime_rocm PUBLIC tvm_runtime ${_rocm_libs})
  set_target_properties(tvm_runtime_rocm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  tvm_set_python_module_relative_rpath(tvm_runtime_rocm)
  install(TARGETS tvm_runtime_rocm DESTINATION lib${LIB_SUFFIX})
  if(TVM_BUILD_PYTHON_MODULE)
    install(TARGETS tvm_runtime_rocm DESTINATION "lib")
  endif()
endif(USE_ROCM)

# HIPBLAS contrib goes into libtvm_runtime_extra.
if(USE_ROCM AND USE_HIPBLAS)
  message(STATUS "Build with HIPBLAS support")
  tvm_file_glob(GLOB HIPBLAS_CONTRIB_SRC src/relax/backend/contrib/hipblas/*.cc)
  list(APPEND COMPILER_SRCS ${HIPBLAS_CONTRIB_SRC})
  tvm_file_glob(GLOB HIPBLAS_CONTRIB_SRCS src/runtime/extra/contrib/hipblas/*.cc)
  add_library(tvm_hipblas_objs OBJECT ${HIPBLAS_CONTRIB_SRCS})
  target_link_libraries(tvm_hipblas_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_hipblas_objs ${ROCM_HIPBLAS_LIBRARY})
  if(NOT ROCM_HIPBLASLT_LIBRARY STREQUAL "ROCM_HIPBLASLT_LIBRARY-NOTFOUND")
    target_link_libraries(tvm_runtime_extra PRIVATE ${ROCM_HIPBLASLT_LIBRARY})
  endif()
endif(USE_ROCM AND USE_HIPBLAS)

if(USE_ROCM AND USE_THRUST)
  message(STATUS "Build with rocThrust support")
  # We need to override CXX to hipcc. This is required by rocthrust
  if(${CMAKE_CXX_COMPILER} MATCHES "hipcc$")
    message(STATUS "Using hipcc compiler to compile rocthrust code.")
  else()
    message(FATAL_ERROR "Set CXX=hipcc to compile rocthrust code.")
  endif()

  find_package(rocprim REQUIRED)
  find_package(rocthrust REQUIRED)
  set_source_files_properties(src/runtime/extra/contrib/thrust/thrust.cu PROPERTIES LANGUAGE CXX)
  add_library(tvm_rocthrust_objs OBJECT src/runtime/extra/contrib/thrust/thrust.cu)
  target_link_libraries(tvm_rocthrust_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_rocthrust_objs roc::rocthrust)
endif(USE_ROCM AND USE_THRUST)
