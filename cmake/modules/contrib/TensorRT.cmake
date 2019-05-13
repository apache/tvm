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

# TensorRT Module

if(IS_DIRECTORY ${USE_TENSORRT})
    set(TENSORRT_ROOT_DIR ${USE_TENSORRT})
    message(STATUS "Custom TensorRT path: " ${TENSORRT_ROOT_DIR})
    set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT_DIR}/include)
    set(TENSORRT_LIB_DIR ${TENSORRT_ROOT_DIR}/lib)
    file(GLOB TENSORRT_SRCS src/contrib/subgraph/*.cc)
    include_directories(${TENSORRT_INCLUDE_DIR})
    list(APPEND RUNTIME_SRCS ${TENSORRT_SRCS})
    find_library(TENSORRT_NVINFER_LIBRARY nvinfer ${TENSORRT_LIB_DIR})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${TENSORRT_NVINFER_LIBRARY})
    set_source_files_properties(${RUNTIME_GRAPH_SRCS}
            PROPERTIES COMPILE_DEFINITIONS "TVM_GRAPH_RUNTIME_TENSORRT")
endif()
