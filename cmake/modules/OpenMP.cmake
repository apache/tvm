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

# OpenMP implements only the core runtime's thread pool.
if(TVM_TIRX_USE_OPENMP STREQUAL "gnu" OR TVM_TIRX_USE_OPENMP STREQUAL "intel")
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_compile_definitions(tvm_runtime_objs PRIVATE TVM_THREADPOOL_USE_OPENMP=1)
    if(TVM_TIRX_USE_OPENMP STREQUAL "gnu")
      target_link_libraries(tvm_runtime_objs PRIVATE OpenMP::OpenMP_CXX)
      target_link_libraries(tvm_runtime PRIVATE OpenMP::OpenMP_CXX)
      target_link_libraries(tvm_runtime_static PRIVATE OpenMP::OpenMP_CXX)
      message(STATUS "Build with OpenMP ${OpenMP_CXX_LIBRARIES}")
    else()
      # Keep Intel's runtime even when the compiler's default OpenMP runtime is GNU.
      if(MSVC)
        find_library(OMP_LIBRARY NAMES libiomp5md)
      else()
        find_library(OMP_LIBRARY NAMES iomp5)
      endif()
      separate_arguments(_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
      target_compile_options(tvm_runtime_objs PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:${_openmp_flags}>")
      target_include_directories(tvm_runtime_objs PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
      target_link_libraries(tvm_runtime PRIVATE ${OMP_LIBRARY})
      target_link_libraries(tvm_runtime_static PRIVATE ${OMP_LIBRARY})
      message(STATUS "Build with OpenMP ${OMP_LIBRARY}")
    endif()
  else()
    target_compile_definitions(tvm_runtime_objs PRIVATE TVM_THREADPOOL_USE_OPENMP=0)
    message(WARNING "OpenMP cannot be found, use TVM threadpool instead.")
  endif()
else()
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_THREADPOOL_USE_OPENMP=0)
endif()
