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

# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(standalone_crt_libs C CXX)

if(NOT DEFINED CRT_CONFIG_PATH)
  message(FATAL_ERROR "Must supply path to crt_config.h: CRT_CONFIG_PATH=...")
endif()

set(CRT_LIB_BASE ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/crt)
if(NOT EXISTS "${CRT_LIB_BASE}")
    message(FATAL_ERROR
    "This CMakeList does not build inside TVM source tree.
Build the standalone_crt target, and re-invoke CMakeList.txt in build/standalone_crt.")
endif()

if (NOT DEFINED CRT_LIBS)
  set(CRT_LIBS microtvm_rpc_server
    microtvm_rpc_common
    aot_executor_module
    aot_executor
    graph_executor_module
    graph_executor
    common
    memory
  )
endif()

foreach(crt_lib_name ${CRT_LIBS})
  add_library(${crt_lib_name})
  file(GLOB_RECURSE crt_lib_srcs ${CRT_LIB_BASE}/${crt_lib_name}/*.c ${CRT_LIB_BASE}/${crt_lib_name}/*.cc)
  target_sources(${crt_lib_name} PRIVATE ${crt_lib_srcs})
  target_include_directories(${crt_lib_name} PRIVATE include ${CRT_CONFIG_PATH})
endforeach(crt_lib_name ${CRT_LIBS})
