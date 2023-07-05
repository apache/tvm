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
cmake_minimum_required(VERSION 3.2)
project(tf_tvmdsoop C CXX)

set(TFTVM_COMPILE_FLAGS -std=c++17)
set(BUILD_TVMDSOOP_ONLY ON)
set(CMAKE_CURRENT_SOURCE_DIR ${TVM_ROOT})
set(CMAKE_CURRENT_BINARY_DIR ${TVM_ROOT}/build)

include_directories(SYSTEM ${TVM_ROOT}/3rdparty/dlpack/include/)
include_directories(SYSTEM ${TVM_ROOT}/3rdparty/dmlc-core/include/)
include_directories(${TVM_ROOT}/include)

link_directories(${TVM_ROOT}/build)

include(${TVM_ROOT}/cmake/util/FindCUDA.cmake)
include(${TVM_ROOT}/cmake/modules/CUDA.cmake)

include(${TVM_ROOT}/cmake/modules/contrib/TF_TVMDSOOP.cmake)
