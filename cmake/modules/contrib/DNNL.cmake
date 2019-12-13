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

if(USE_DNNL_CODEGEN STREQUAL "ON")
  file(GLOB DNNL_RELAY_CONTRIB_SRC src/relay/backend/contrib/dnnl/codegen.cc)
  list(APPEND COMPILER_SRCS ${DNNL_RELAY_CONTRIB_SRC})

  find_library(EXTERN_LIBRARY_DNNL dnnl)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_DNNL})
  file(GLOB DNNL_CONTRIB_SRC src/runtime/contrib/dnnl/*)
  list(APPEND RUNTIME_SRCS ${DNNL_CONTRIB_SRC})
  message(STATUS "Build with DNNL codegen: " ${EXTERN_LIBRARY_DNNL})
endif()

