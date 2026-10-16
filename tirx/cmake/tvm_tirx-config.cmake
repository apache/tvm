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

include(CMakeFindDependencyMacro)
find_dependency(Python 3.10 COMPONENTS Interpreter)

# Query this installation's helper, independently of any other Python wheel.
# Passing the exact library directory also supports lib64 and multiarch layouts.
get_filename_component(_tvm_tirx_libdir "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
execute_process(
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/config.py"
    --library-dir "${_tvm_tirx_libdir}" --platform "${CMAKE_SYSTEM_NAME}" --json
  OUTPUT_VARIABLE _tvm_tirx_config OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(JSON TVM_TIRX_VERSION GET "${_tvm_tirx_config}" version)
string(JSON tvm_tirx_INCLUDE_DIR GET "${_tvm_tirx_config}" include_dir)
set(tvm_tirx_VERSION "${TVM_TIRX_VERSION}")
set(TVM_TIRX_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

if(NOT TARGET tvm_tirx::header)
  add_library(tvm_tirx::header INTERFACE IMPORTED)
  target_compile_features(tvm_tirx::header INTERFACE cxx_std_20)
  target_compile_definitions(tvm_tirx::header INTERFACE TVM_INDEX_DEFAULT_I64=1)
  target_include_directories(tvm_tirx::header INTERFACE "${tvm_tirx_INCLUDE_DIR}")
endif()

foreach(_library tvm_runtime tvm_tirx)
  if(TARGET ${_library}::shared)
    continue()
  endif()
  string(JSON _location GET "${_tvm_tirx_config}" libraries ${_library} location)
  add_library(${_library}::shared SHARED IMPORTED)
  set_target_properties(${_library}::shared PROPERTIES IMPORTED_LOCATION "${_location}")
  if(WIN32)
    string(JSON _implib GET "${_tvm_tirx_config}" libraries ${_library} implib)
    set_target_properties(${_library}::shared PROPERTIES IMPORTED_IMPLIB "${_implib}")
  endif()
  target_link_libraries(${_library}::shared INTERFACE tvm_tirx::header)
  if(_library STREQUAL "tvm_tirx")
    target_link_libraries(tvm_tirx::shared INTERFACE tvm_runtime::shared)
  endif()
endforeach()
