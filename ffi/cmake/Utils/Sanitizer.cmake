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

function(add_sanitizer_address target_name)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    include(CheckCXXCompilerFlag)
    set (_saved_CRF ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
    check_cxx_source_compiles("int main() { return 0; }" COMPILER_SUPPORTS_ASAN)
    set (CMAKE_REQUIRED_FLAGS ${_saved_CRF})
    get_target_property(_saved_type ${target_name} TYPE)
    if (${_saved_type} STREQUAL "INTERFACE_LIBRARY")
      set(_saved_type INTERFACE)
    else()
      set(_saved_type PRIVATE)
    endif()
    target_link_options(${target_name} ${_saved_type} "-fsanitize=address")
    target_compile_options(${target_name} ${_saved_type} "-fsanitize=address" "-fno-omit-frame-pointer" "-g")
    return()
  endif()
endfunction()
