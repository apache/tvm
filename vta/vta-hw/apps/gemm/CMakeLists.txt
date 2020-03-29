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
project(tsim C CXX)

if(NOT DEFINED ENV{TVM_PATH})
    message(ERROR "Make sure to set TVM_PATH in your environment")
endif()

if(NOT DEFINED ENV{VTA_HW_PATH})
    message(ERROR "Make sure to set VTA_HW_PATH in your environment")
endif()

include_directories("$ENV{TVM_PATH}/include")
include_directories("$ENV{TVM_PATH}/3rdparty/dlpack/include")
include_directories("$ENV{TVM_PATH}/3rdparty/dmlc-core/include")
include_directories("$ENV{VTA_HW_PATH}/src/dpi")

set(CMAKE_C_FLAGS "-O2 -Wall -fPIC -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -fvisibility=hidden -std=c++11")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
endif()

file(GLOB TSIM_SW_SRC src/driver.cc)
list(APPEND TSIM_SW_SRC $ENV{VTA_HW_PATH}/src/vmem/virtual_memory.cc)
list(APPEND TSIM_SW_SRC $ENV{VTA_HW_PATH}/src/dpi/module.cc)

add_library(sw SHARED ${TSIM_SW_SRC})
target_include_directories(sw PRIVATE $ENV{VTA_HW_PATH}/include $ENV{VTA_HW_PATH}/src)

if(APPLE)
  set_target_properties(sw PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif(APPLE)
