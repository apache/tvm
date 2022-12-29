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

if(USE_BNNS STREQUAL "ON")
  add_definitions(-DUSE_JSON_RUNTIME=1)
  tvm_file_glob(GLOB BNNS_RELAY_CONTRIB_SRC src/relay/backend/contrib/bnns/*.cc)
  list(APPEND COMPILER_SRCS ${BNNS_RELAY_CONTRIB_SRC})
  list(APPEND COMPILER_SRCS ${JSON_RELAY_CONTRIB_SRC})

  list(APPEND TVM_RUNTIME_LINKER_LIBS "-framework Accelerate")

  tvm_file_glob(GLOB BNNS_CONTRIB_SRC src/runtime/contrib/bnns/*.cc)
  list(APPEND RUNTIME_SRCS ${BNNS_CONTRIB_SRC})
  message(STATUS "Build with BNNS JSON runtime: " ${EXTERN_LIBRARY_BNNS})
endif()
