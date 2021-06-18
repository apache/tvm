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

if (USE_MLAS)
    message(STATUS "Build with MLAS library")
    if (NOT (USE_OPENMP STREQUAL "gnu" OR USE_OPENMP STREQUAL "intel"))
        message(FATAL_ERROR "MLAS library must be built with USE_OPENMP=gnu or USE_OPENMP=intel")
    endif()
    add_subdirectory("3rdparty/mlas")
    list(APPEND RUNTIME_SRCS src/runtime/contrib/mlas/mlas_op.cc)
    list(APPEND TVM_RUNTIME_LINKER_LIBS onnxruntime_mlas_static)
    include_directories(SYSTEM "3rdparty/mlas/inc")
endif()
