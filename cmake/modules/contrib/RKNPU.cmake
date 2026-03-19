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

# RKNPU backend: codegen is in Python (python/tvm/relax/backend/contrib/rknpu/),
# so USE_RKNPU_CODEGEN only needs to compile the C++ runtime into libtvm.
if(USE_RKNPU_CODEGEN)
    message(STATUS "Build with RKNPU codegen")

    tvm_file_glob(GLOB RUNTIME_RKNPU_SRCS src/runtime/contrib/rknpu/*.cc)
    if(NOT USE_RKNPU_RUNTIME)
        list(APPEND COMPILER_SRCS ${RUNTIME_RKNPU_SRCS})
    endif()
endif()

# RKNPU Runtime
if(USE_RKNPU_RUNTIME)
    message(STATUS "Build with RKNPU runtime")

    tvm_file_glob(GLOB RUNTIME_RKNPU_SRCS src/runtime/contrib/rknpu/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_RKNPU_SRCS})

    add_definitions(-DTVM_RKNPU_RUNTIME)
endif()
