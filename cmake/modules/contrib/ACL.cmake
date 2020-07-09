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

# We separate the codegen and runtime build since ACL can only be built
# for AArch. In the world where we take the cross compilation approach,
# which is common with arm devices, we need to be able to cross-compile
# a relay graph on x86 for AArch and then run the graph on AArch.
if(USE_ACL)
    file(GLOB ACL_RELAY_CONTRIB_SRC src/relay/backend/contrib/acl/*.cc)
    file(GLOB ACL_RUNTIME_MODULE src/runtime/contrib/acl/acl_runtime.cc)
    list(APPEND COMPILER_SRCS ${ACL_RELAY_CONTRIB_SRC})
    list(APPEND COMPILER_SRCS ${ACL_RUNTIME_MODULE})
    message(STATUS "Build with ACL support...")
endif()

if(USE_ACL_GRAPH_RUNTIME)
    set(ACL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/acl)
    # Detect custom ACL path.
    if (NOT USE_ACL_GRAPH_RUNTIME STREQUAL "ON")
        set(ACL_PATH ${USE_ACL_GRAPH_RUNTIME})
    endif()

    file(GLOB ACL_CONTRIB_SRC src/runtime/contrib/acl/*)
    file(GLOB ACL_API src/relay/backend/contrib/acl/acl_api.cc)

    set(ACL_INCLUDE_DIRS ${ACL_PATH}/include ${ACL_PATH})
    include_directories(${ACL_INCLUDE_DIRS})

    find_library(EXTERN_ACL_COMPUTE_LIB
            NAMES arm_compute libarm_compute
            HINTS "${ACL_PATH}" "${ACL_PATH}/lib" "${ACL_PATH}/build"
            )
    find_library(EXTERN_ACL_COMPUTE_CORE_LIB
            NAMES arm_compute_core libarm_compute_core
            HINTS "${ACL_PATH}" "${ACL_PATH}/lib" "${ACL_PATH}/build"
            )
    find_library(EXTERN_ACL_COMPUTE_GRAPH_LIB
            NAMES arm_compute_graph libarm_compute_graph
            HINTS "${ACL_PATH}" "${ACL_PATH}/lib" "${ACL_PATH}/build"
            )

    list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_ACL_COMPUTE_LIB})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_ACL_COMPUTE_CORE_LIB})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_ACL_COMPUTE_GRAPH_LIB})
    list(APPEND RUNTIME_SRCS ${ACL_CONTRIB_SRC})
    list(APPEND RUNTIME_SRCS ${ACL_API})
    message(STATUS "Build with ACL graph runtime support: "
            ${EXTERN_ACL_COMPUTE_LIB} ", \n"
            ${EXTERN_ACL_COMPUTE_CORE_LIB} ", \n"
            ${EXTERN_ACL_COMPUTE_GRAPH_LIB})

    # Set flag to detect ACL graph runtime support.
    add_definitions(-DTVM_GRAPH_RUNTIME_ACL)
endif()
