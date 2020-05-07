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

project(SIM_DEV C CXX)
cmake_minimum_required(VERSION 3.0.2)

set(CMAKE_SYSTEM_NAME "Linux")

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
endif()

set(EXTRA_CXX_FLAGS
  "-O2"
  "-Wno-format"
  "-mhvx -mhvx-length=128b"
  "-mv60"
  "-stdlib=libc++"
)

set(EXTRA_LINK_FLAGS
  "-stdlib=libc++"
  "-G0"
  "-Wl,--force-dynamic"
  "-Wl,--export-dynamic"
  "-Wl,--whole-archive"   # This should link entire libc, libc++ and libc+abi.
  "-Wl,--defsym=HEAP_SIZE=0x40000000"
)

string(REGEX REPLACE ";" " " EXTRA_CXX_FLAGS_STR "${EXTRA_CXX_FLAGS}")
string(REGEX REPLACE ";" " " EXTRA_LINK_FLAGS_STR "${EXTRA_LINK_FLAGS}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${EXTRA_CXX_FLAGS_STR} ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${EXTRA_LINK_FLAGS_STR} ${CMAKE_EXE_LINKER_FLAGS}")

# Set project properties.

file(GLOB SOURCE_FILES "*.cc")
add_executable(sim_dev ${SOURCE_FILES})
target_include_directories(sim_dev
  PUBLIC "."
  PUBLIC ".."
  PUBLIC "../../../../../include"
  PUBLIC "../../../../../3rdparty/dlpack/include"
)

target_link_libraries(sim_dev "-ldl")
