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

# TensorRT Runtime
if(USE_TENSORRT)
    # Enable codegen as well
    SET(USE_TENSORRT_CODEGEN ON)
    if(IS_DIRECTORY ${USE_TENSORRT})
        set(TENSORRT_ROOT_DIR ${USE_TENSORRT})
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

    # NNVM TRT runtime sources
    file(GLOB TENSORRT_NNVM_SRCS src/contrib/subgraph/*.cc)
    list(APPEND RUNTIME_SRCS ${TENSORRT_NNVM_SRCS})

    # Relay TRT runtime sources
    file(GLOB TENSORRT_RELAY_CONTRIB_SRC src/runtime/contrib/tensorrt/*.cc)
    list(APPEND RUNTIME_SRCS ${TENSORRT_RELAY_CONTRIB_SRC})
    list(APPEND RUNTIME_SRCS src/relay/backend/contrib/tensorrt/common_utils.cc)

    # Set defines
    set_source_files_properties(${RUNTIME_GRAPH_SRCS}
            PROPERTIES COMPILE_DEFINITIONS "TVM_GRAPH_RUNTIME_TENSORRT")
endif()
# TensorRT Codegen only. This can be enabled independently of USE_TENSORRT to
# enable compilation of TensorRT modules without requiring TensorRT to be
# installed. The compiled modules will only be able to be executed using a TVM
# built with USE_TENSORRT=ON.
if(USE_TENSORRT_CODEGEN)
    message(STATUS "Build with TensorRT codegen")
    # Relay TRT codegen sources
    file(GLOB TENSORRT_RELAY_CONTRIB_SRC src/relay/backend/contrib/tensorrt/*.cc)
    list(APPEND COMPILER_SRCS ${TENSORRT_RELAY_CONTRIB_SRC})
    list(APPEND COMPILER_SRCS src/runtime/contrib/tensorrt/tensorrt_module.cc)
    # If runtime is enabled also, set flag for compiler srcs
    if(USE_TENSORRT)
        set_source_files_properties(${COMPILER_SRCS}
                PROPERTIES COMPILE_DEFINITIONS "TVM_GRAPH_RUNTIME_TENSORRT")
    endif()
endif()
