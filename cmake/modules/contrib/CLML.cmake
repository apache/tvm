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

if(USE_CLML)
    file(GLOB CLML_RELAY_CONTRIB_SRC src/relay/backend/contrib/clml/*.cc)
    file(GLOB CLML_RUNTIME_MODULE src/runtime/contrib/clml/clml_runtime.cc)
    include_directories(SYSTEM "3rdparty/OpenCL-Headers")
    list(APPEND COMPILER_SRCS ${CLML_RELAY_CONTRIB_SRC})
    if(NOT USE_CLML_GRAPH_EXECUTOR)
        list(APPEND COMPILER_SRCS ${CLML_RUNTIME_MODULE})
    endif()
    message(STATUS "Build with CLML support : " ${USE_CLML})
    if (NOT USE_CLML STREQUAL "ON")
        set(CLML_VERSION_HEADER "${USE_CLML}/CL/cl_qcom_ml_ops.h")
        if(EXISTS ${CLML_VERSION_HEADER})
            file(READ ${CLML_VERSION_HEADER} ver)
            string(REGEX MATCH "CL_QCOM_ML_OPS_H_MAJOR_VERSION ([0-9]*)" _ ${ver})
            set(CLML_VERSION_MAJOR ${CMAKE_MATCH_1})
        else()
            set(CLML_VERSION_MAJOR "2")
        endif()
    else()
        set(CLML_VERSION_MAJOR "2")
    endif()
    add_definitions(-DTVM_CLML_VERSION=${CLML_VERSION_MAJOR})
    message(STATUS "CLML SDK Version :" ${CLML_VERSION_MAJOR})
endif()

if(USE_CLML_GRAPH_EXECUTOR)
    set(CLML_PATH ${CMAKE_CURRENT_SOURCE_DIR}/clml)
    # Detect custom CLML path.
    if (NOT USE_CLML_GRAPH_EXECUTOR STREQUAL "ON")
        set(CLML_PATH ${USE_CLML_GRAPH_EXECUTOR})
    endif()

    file(GLOB CLML_CONTRIB_SRC src/runtime/contrib/clml/*)

    # Cmake needs to find clml library, include and support directories
    # in the path specified by CLML_PATH.
    set(CLML_INCLUDE_DIRS ${CLML_PATH}/include ${CLML_PATH})
    include_directories(${CLML_INCLUDE_DIRS})
    find_library(EXTERN_CLML_COMPUTE_LIB
          NAMES OpenCL libOpenCL
          HINTS "${CLML_PATH}" "${CLML_PATH}/lib64" "${CLML_PATH}/lib"
          )
    if(NOT EXTERN_CLML_COMPUTE_LIB)
        string(FIND ${ANDROID_ABI} "64" ARCH_64)
        set(EXTERN_CLML_COMPUTE_LIB "")
        if(ARCH_64 GREATER -1)
            list(APPEND EXTERN_CLML_COMPUTE_LIB ${CLML_PATH}/lib64/libOpenCL.so ${CLML_PATH}/lib64/libOpenCL_system.so)
        else()
            list(APPEND EXTERN_CLML_COMPUTE_LIB ${CLML_PATH}/lib/libOpenCL.so ${CLML_PATH}/lib/libOpenCL_system.so)
        endif()
    endif()
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_CLML_COMPUTE_LIB})
    list(APPEND RUNTIME_SRCS ${CLML_CONTRIB_SRC})
    message(STATUS "Build with CLML graph runtime support: "
            ${EXTERN_CLML_COMPUTE_LIB})

    # Set flag to detect CLML graph runtime support.
    add_definitions(-DTVM_GRAPH_EXECUTOR_CLML)

    message(STATUS "Enable OpenCL as fallback to CLML")
    file(GLOB RUNTIME_OPENCL_SRCS src/runtime/opencl/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_OPENCL_SRCS})
    set(USE_OPENCL ${CLML_PATH})
    if(USE_OPENCL_ENABLE_HOST_PTR)
        add_definitions(-DOPENCL_ENABLE_HOST_PTR)
    endif(USE_OPENCL_ENABLE_HOST_PTR)
endif()
