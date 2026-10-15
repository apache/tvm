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

# Target-local policies shared by the compiler and runtime backends.
function(tvm_tirx_configure_target target)
  set_target_properties(${target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
  target_compile_definitions(${target} PRIVATE TVM_VERSION="${TVM_TIRX_VERSION}")
  target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/3rdparty/compiler-rt")
  if(MSVC)
    target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
      _SCL_SECURE_NO_WARNINGS _ENABLE_EXTENDED_ALIGNED_STORAGE NOMINMAX)
    target_compile_options(${target} PRIVATE /EHsc /MP /bigobj /permissive-
      /wd4244 /wd4267 /wd4018 /wd4316 /wd4101 /wd4180 /wd4251 /wd4624 /wd4146 /wd4141 /wd4068)
  else()
    target_compile_options(${target} PRIVATE -Wall)
  endif()
  tvm_ffi_add_prefix_map(${target} "${PROJECT_SOURCE_DIR}")
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "SHARED_LIBRARY")
    if(APPLE)
      target_link_options(${target} PRIVATE "-Wl,-undefined,error")
      set_target_properties(${target} PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
    elseif(UNIX)
      target_link_options(${target} PRIVATE "-Wl,--no-undefined")
      set_target_properties(${target} PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
    endif()
    tvm_ffi_hide_static_linked_lib_symbols(${target})
    tvm_ffi_add_apple_dsymutil(${target})
  endif()
  # Honor a parent's output layout; otherwise keep standalone libraries together.
  foreach(_kind ARCHIVE LIBRARY RUNTIME)
    if(DEFINED CMAKE_${_kind}_OUTPUT_DIRECTORY)
      set(_output "${CMAKE_${_kind}_OUTPUT_DIRECTORY}")
    else()
      set(_output "${PROJECT_BINARY_DIR}/lib")
    endif()
    set_target_properties(${target} PROPERTIES ${_kind}_OUTPUT_DIRECTORY "${_output}")
    foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
      string(TOUPPER "${_config}" _config)
      set_target_properties(${target} PROPERTIES ${_kind}_OUTPUT_DIRECTORY_${_config} "${_output}")
    endforeach()
  endforeach()
endfunction()
