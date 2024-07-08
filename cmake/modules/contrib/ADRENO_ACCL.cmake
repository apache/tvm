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

if (USE_ADRENO_ACCL)
    file(GLOB ADRENO_ACCL_RELAY_CONTRIB_SRC src/relax/backend/contrib/adreno_accl/*.cc)
    file(GLOB ADRENO_ACCL_RUNTIME_MODULE src/runtime/contrib/adreno_accl/adreno_accl_runtime.cc)
    list(APPEND COMPILER_SRCS ${ADRENO_ACCL_RELAY_CONTRIB_SRC})
    if(NOT USE_ADRENO_ACCL_GRAPH_EXECUTOR)
        list(APPEND COMPILER_SRCS ${ADRENO_ACCL_RUNTIME_MODULE})
    endif()
    message(STATUS "Build with ADRENO_ACCL support : " ${USE_ADRENO_ACCL})
    if (NOT USE_ADRENO_ACCL STREQUAL "ON")
        set(ADRENO_ACCL_VERSION_HEADER "${USE_ADRENO_ACCL}/include/adrenoaccl.h")
        if(EXISTS ${ADRENO_ACCL_VERSION_HEADER})
            file(READ ${ADRENO_ACCL_VERSION_HEADER} ver)
            string(REGEX MATCH "ADRENO_ACCL_H_MAJOR_VERSION ([0-9]*)" _ ${ver})
            set(ADRENO_ACCL_VERSION_MAJOR ${CMAKE_MATCH_1})
        else()
            set(ADRENO_ACCL_VERSION_MAJOR "1")
        endif()
    else()
        set(ADRENO_ACCL_VERSION_MAJOR "1")
    endif()
    add_definitions(-DTVM_ADRENO_ACCL_VERSION=${ADRENO_ACCL_VERSION_MAJOR})
    message(STATUS "ADRENO_ACCL SDK Version :" ${ADRENO_ACCL_VERSION_MAJOR})
endif()

if(USE_ADRENO_ACCL_GRAPH_EXECUTOR)
    set(ADRENO_ACCL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/adrenoaccl)
    # Detect custom ADRENO_ACCL path.
    if (NOT USE_ADRENO_ACCL_GRAPH_EXECUTOR STREQUAL "ON")
        set(ADRENO_ACCL_PATH ${USE_ADRENO_ACCL_GRAPH_EXECUTOR})
    endif()

    file(GLOB ADRENO_ACCL_CONTRIB_SRC src/runtime/contrib/adreno_accl/*)

    # Cmake needs to find adreno_accl library, include and support directories
    # in the path specified by ADRENO_ACCL_PATH.
    set(ADRENO_ACCL_INCLUDE_DIRS ${ADRENO_ACCL_PATH}/include ${ADRENO_ACCL_PATH})
    include_directories(${ADRENO_ACCL_INCLUDE_DIRS})
    find_library(EXTERN_ADRENO_ACCL_COMPUTE_LIB
         NAMES adrenoaccl
         HINTS "${ADRENO_ACCL_PATH}" "${ADRENO_ACCL_PATH}/lib"
         )
    if(NOT EXTERN_ADRENO_ACCL_COMPUTE_LIB)
        set(EXTERN_ADRENO_ACCL_COMPUTE_LIB "")
        list(APPEND EXTERN_ADRENO_ACCL_COMPUTE_LIB ${ADRENO_ACCL_PATH}/lib/libadrenoaccl.so)
    endif()
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_ADRENO_ACCL_COMPUTE_LIB})
    list(APPEND RUNTIME_SRCS ${ADRENO_ACCL_CONTRIB_SRC})
    message(STATUS "Build with ADRENO_ACCL graph runtime support: "
            ${EXTERN_ADRENO_ACCL_COMPUTE_LIB})

    # Set flag to detect ADRENO_ACCL graph runtime support.
    add_definitions(-DTVM_GRAPH_EXECUTOR_ADRENO_ACCL)

    message(STATUS "Enable OpenCL as fallback to ADRENO_ACCL")
    file(GLOB RUNTIME_OPENCL_SRCS src/runtime/opencl/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_OPENCL_SRCS})
    set(USE_OPENCL ON)
    if(USE_OPENCL_ENABLE_HOST_PTR)
        add_definitions(-DOPENCL_ENABLE_HOST_PTR)
    endif(USE_OPENCL_ENABLE_HOST_PTR)
endif()
