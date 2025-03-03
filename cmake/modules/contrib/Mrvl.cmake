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
include(ExternalProject)
if(USE_MRVL)
  # Mrvl Module
  message(STATUS "Build with Mrvl support")
  file(GLOB RUNTIME_MRVL_SRCS
    src/runtime/contrib/mrvl/mrvl_runtime.cc
    src/runtime/contrib/mrvl/mrvl_hw_runtime.cc
    src/runtime/contrib/mrvl/mrvl_sw_runtime_lib.cc
  )
  list(APPEND RUNTIME_SRCS ${RUNTIME_MRVL_SRCS})
  file(GLOB COMPILER_MRVL_SRCS
    src/relay/backend/contrib/mrvl/codegen.cc
    src/relay/backend/contrib/mrvl/compiler_attr.cc
  )
  list(APPEND COMPILER_SRCS ${COMPILER_MRVL_SRCS})
endif(USE_MRVL)
