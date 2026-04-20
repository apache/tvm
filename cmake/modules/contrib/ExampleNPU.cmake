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

# Example NPU Codegen
if(USE_EXAMPLE_NPU_CODEGEN)
    message(STATUS "Build with Example NPU codegen")

    tvm_file_glob(GLOB COMPILER_EXAMPLE_NPU_SRCS src/relax/backend/contrib/example_npu/*.cc)
    list(APPEND COMPILER_SRCS ${COMPILER_EXAMPLE_NPU_SRCS})

    tvm_file_glob(GLOB RUNTIME_EXAMPLE_NPU_SRCS src/runtime/contrib/example_npu/*.cc)
    if(NOT USE_EXAMPLE_NPU_RUNTIME)
        list(APPEND COMPILER_SRCS ${RUNTIME_EXAMPLE_NPU_SRCS})
    endif()
endif()

# Example NPU Runtime
if(USE_EXAMPLE_NPU_RUNTIME)
    message(STATUS "Build with Example NPU runtime")

    tvm_file_glob(GLOB RUNTIME_EXAMPLE_NPU_SRCS src/runtime/contrib/example_npu/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_EXAMPLE_NPU_SRCS})

    add_definitions(-DTVM_GRAPH_EXECUTOR_EXAMPLE_NPU)
endif()
