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

if(USE_SDACCEL)
  message(STATUS "Build with SDAccel support")
  tvm_file_glob(GLOB RUNTIME_SDACCEL_SRCS src/runtime/opencl/sdaccel/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_SDACCEL_SRCS})
  if(NOT USE_OPENCL)
    message(STATUS "Enable OpenCL support required for SDAccel")
    set(USE_OPENCL ON)
  endif()
else()
  list(APPEND COMPILER_SRCS src/target/opt/build_sdaccel_off.cc)
endif(USE_SDACCEL)

if(USE_AOCL)
  message(STATUS "Build with Intel FPGA SDK for OpenCL support")
  tvm_file_glob(GLOB RUNTIME_AOCL_SRCS src/runtime/opencl/aocl/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_AOCL_SRCS})
  if(NOT USE_OPENCL)
    message(STATUS "Enable OpenCL support required for Intel FPGA SDK for OpenCL")
    set(USE_OPENCL ON)
  endif()
else()
  list(APPEND COMPILER_SRCS src/target/opt/build_aocl_off.cc)
endif(USE_AOCL)

if(USE_OPENCL)
  tvm_file_glob(GLOB RUNTIME_OPENCL_SRCS src/runtime/opencl/*.cc)

  if(${USE_OPENCL} MATCHES ${IS_TRUE_PATTERN})
    message(WARNING "Build with OpenCL wrapper")
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

  if(DEFINED USE_OPENCL_GTEST AND EXISTS ${USE_OPENCL_GTEST})
    file_glob_append(RUNTIME_OPENCL_SRCS
      "${CMAKE_SOURCE_DIR}/tests/cpp-runtime/opencl/*.cc"
    )
  endif()
  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENCL_SRCS})
else()
  list(APPEND COMPILER_SRCS src/target/opt/build_opencl_off.cc)
endif(USE_OPENCL)
