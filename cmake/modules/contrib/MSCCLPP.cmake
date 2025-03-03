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

if(USE_CUDA AND USE_NCCL AND USE_MSCCL)
  include(FetchContent)
  FetchContent_Declare(
    mscclpp
    GIT_REPOSITORY https://github.com/csullivan/mscclpp.git
    GIT_TAG feature/2024-03-19/msccl-nccl-equivalents
  )
  set(USE_CUDA ON)
  set(BYPASS_PEERMEM_CHECK ON)
  set(BUILD_PYTHON_BINDINGS OFF)
  set(BUILD_TESTS OFF)
  FetchContent_MakeAvailable(mscclpp)

  tvm_file_glob(GLOB MSCCL_SRCS
    ${PROJECT_SOURCE_DIR}/src/runtime/contrib/mscclpp/*.cu
  )

  add_library(msccl SHARED ${MSCCL_SRCS})
  target_link_libraries(msccl PUBLIC mscclpp)
  target_compile_definitions(msccl PRIVATE DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
  target_include_directories(msccl PUBLIC
    $<BUILD_INTERFACE:${mscclpp_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rdparty/mscclpp/include>
    $<INSTALL_INTERFACE:include/msccl>
  )

  install(TARGETS mscclpp_obj
    EXPORT ${PROJECT_NAME}Targets
    FILE_SET HEADERS DESTINATION ${INSTALL_PREFIX}/include)
  install(TARGETS mscclpp EXPORT ${PROJECT_NAME}Targets DESTINATION lib${LIB_SUFFIX})
  install(TARGETS msccl EXPORT ${PROJECT_NAME}Targets DESTINATION lib${LIB_SUFFIX})
  list(APPEND TVM_RUNTIME_LINKER_LIBS msccl)
endif()
