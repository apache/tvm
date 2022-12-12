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

# Arm(R) Ethos(TM)-N rules

if(NOT USE_ETHOSN STREQUAL "OFF")
  find_ethosn(${USE_ETHOSN})

  if(NOT ETHOSN_FOUND)
    message(FATAL_ERROR "Cannot find Arm(R) Ethos(TM)-N, USE_ETHOSN=" ${USE_ETHOSN})

  else()
    include_directories(SYSTEM ${ETHOSN_INCLUDE_DIRS})
    add_definitions(${ETHOSN_DEFINITIONS})

    message(STATUS "Build with Arm(R) Ethos(TM)-N ${ETHOSN_PACKAGE_VERSION}")

    tvm_file_glob(GLOB ETHOSN_RUNTIME_CONTRIB_SRC
                  src/runtime/contrib/ethosn/ethosn_runtime.cc
                  src/runtime/contrib/ethosn/ethosn_device.cc)
    list(APPEND RUNTIME_SRCS ${ETHOSN_RUNTIME_CONTRIB_SRC})

    tvm_file_glob(GLOB COMPILER_ETHOSN_SRCS
                  src/relay/backend/contrib/ethosn/*
                  src/relay/backend/contrib/constant_transforms.cc)
    list(APPEND COMPILER_SRCS ${COMPILER_ETHOSN_SRCS})

    list(APPEND TVM_LINKER_LIBS ${ETHOSN_COMPILER_LIBRARY}
      ${ETHOSN_RUNTIME_LIBRARY})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${ETHOSN_RUNTIME_LIBRARY})

    if(NOT MSVC)
      set_source_files_properties(${COMPILER_ETHOSN_SRCS}
        PROPERTIES COMPILE_DEFINITIONS "DMLC_ENABLE_RTTI=0")
      set_source_files_properties(${COMPILER_ETHOSN_SRCS}
        PROPERTIES COMPILE_FLAGS "-fno-rtti")
    endif()
  endif(NOT ETHOSN_FOUND)
else()
  if(USE_ETHOSN_HW)
    message(FATAL_ERROR "Cannot enable Arm(R) Ethos(TM)-N HW if USE_ETHOSN=OFF")
  endif()
endif(NOT USE_ETHOSN STREQUAL "OFF")
