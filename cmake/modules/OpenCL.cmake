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

if(USE_OPENCL)
  tvm_file_glob(GLOB RUNTIME_OPENCL_SRCS src/runtime/opencl/*.cc)

  if(${USE_OPENCL} MATCHES ${IS_TRUE_PATTERN})
    message(STATUS "Enabled runtime search for OpenCL library location")
    file_glob_append(RUNTIME_OPENCL_SRCS
      "src/runtime/opencl/opencl_wrapper/opencl_wrapper.cc"
    )
    include_directories(SYSTEM "3rdparty/OpenCL-Headers")
  else()
    find_opencl(${USE_OPENCL})
    if(NOT OpenCL_FOUND)
        message(FATAL_ERROR "Error! Cannot find specified OpenCL library")
    endif()
    message(STATUS "Build with OpenCL support")
    include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${OpenCL_LIBRARIES})
  endif()

  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENCL_SRCS})
  if(USE_OPENCL_ENABLE_HOST_PTR)
    add_definitions(-DOPENCL_ENABLE_HOST_PTR)
  endif(USE_OPENCL_ENABLE_HOST_PTR)
  if(USE_OPENCL_EXTN_QCOM)
    add_definitions(-DUSE_OPENCL_EXTN_QCOM)
    find_path(ocl_header cl.h HINTS ${OpenCL_INCLUDE_DIRS} PATH_SUFFIXES CL)
    set(OCL_VERSION_HEADER "${ocl_header}/cl.h")
    if(EXISTS ${OCL_VERSION_HEADER})
      file(READ ${OCL_VERSION_HEADER} ver)
      string(REGEX MATCH "CL_TARGET_OPENCL_VERSION ([0-9]*)" _ ${ver})
      add_definitions(-DCL_TARGET_OPENCL_VERSION=${CMAKE_MATCH_1})
      message(STATUS "Set OpenCL Target version to " ${CMAKE_MATCH_1})
    endif()
  endif(USE_OPENCL_EXTN_QCOM)
endif(USE_OPENCL)
