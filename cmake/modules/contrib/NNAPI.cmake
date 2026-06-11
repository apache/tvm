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

# NNAPI Codegen
if(USE_NNAPI_CODEGEN)
    message(STATUS "Build with NNAPI codegen")

    tvm_file_glob(GLOB COMPILER_NNAPI_SRCS src/relax/backend/contrib/nnapi/*.cc)
    tvm_file_glob(GLOB RUNTIME_NNAPI_SRCS src/runtime/extra/contrib/nnapi/*.cc)
    list(APPEND COMPILER_SRCS ${COMPILER_NNAPI_SRCS})
    if(NOT USE_NNAPI_RUNTIME)
        list(APPEND COMPILER_SRCS ${RUNTIME_NNAPI_SRCS})
    endif()
endif()

# NNAPI Runtime — goes into libtvm_runtime_extra.
if(USE_NNAPI_RUNTIME)
    message(STATUS "Build with NNAPI runtime")

    tvm_file_glob(GLOB RUNTIME_NNAPI_SRCS src/runtime/extra/contrib/nnapi/*.cc)
    add_library(tvm_nnapi_objs OBJECT ${RUNTIME_NNAPI_SRCS})
    target_link_libraries(tvm_nnapi_objs PRIVATE tvm_runtime_extra_defs)
    target_link_libraries(tvm_runtime_extra PRIVATE tvm_nnapi_objs neuralnetworks log)

    add_definitions(-DTVM_GRAPH_EXECUTOR_NNAPI)
endif()
