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

# TensorRT Codegen only. This can be enabled independently of USE_TENSORRT_RUNTIME to enable
# compilation of TensorRT modules without requiring TensorRT to be installed. The compiled modules
# will only be able to be executed using a TVM built with USE_TENSORRT_RUNTIME=ON.

include (FindPackageHandleStandardArgs)

if(USE_TENSORRT_CODEGEN)
    message(STATUS "Build with TensorRT codegen")
    tvm_file_glob(GLOB COMPILER_TENSORRT_SRCS src/relay/backend/contrib/tensorrt/*.cc src/relax/backend/contrib/tensorrt/*.cc)
    set_source_files_properties(${COMPILER_TENSORRT_SRCS} PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
    tvm_file_glob(GLOB RUNTIME_TENSORRT_SRCS src/runtime/contrib/tensorrt/tensorrt_runtime.cc)
    set_source_files_properties(${RUNTIME_TENSORRT_SRCS} PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
    list(APPEND COMPILER_SRCS ${COMPILER_TENSORRT_SRCS})
    if(NOT USE_TENSORRT_RUNTIME)
        list(APPEND COMPILER_SRCS ${RUNTIME_TENSORRT_SRCS})
    endif()
endif()

# TensorRT Runtime
if(USE_TENSORRT_RUNTIME)
    if(IS_DIRECTORY ${USE_TENSORRT_RUNTIME})
        set(TENSORRT_ROOT_DIR ${USE_TENSORRT_RUNTIME})
        message(STATUS "Custom TensorRT path: " ${TENSORRT_ROOT_DIR})
    endif()
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES include)
    find_library(TENSORRT_LIB_DIR nvinfer HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES lib)
    find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIB_DIR)
    if(NOT TENSORRT_FOUND)
        message(ERROR "Could not find TensorRT.")
    endif()
    message(STATUS "TENSORRT_LIB_DIR: " ${TENSORRT_LIB_DIR})
    include_directories(${TENSORRT_INCLUDE_DIR})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${TENSORRT_LIB_DIR})

    # TRT runtime sources
    tvm_file_glob(GLOB RUNTIME_TENSORRT_SRCS src/runtime/contrib/tensorrt/*.cc)
    set_source_files_properties(${RUNTIME_TENSORRT_SRCS} PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
    list(APPEND RUNTIME_SRCS ${RUNTIME_TENSORRT_SRCS})

    # Set defines
    add_definitions(-DTVM_GRAPH_EXECUTOR_TENSORRT)
endif()
