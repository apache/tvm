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

# Query the helper belonging to this installation, even when Python can see a
# different TIRx wheel. The library directory also supports lib64/multiarch.
get_filename_component(_tvm_tirx_libdir "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
execute_process(
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/config.py"
    --library-dir "${_tvm_tirx_libdir}" --platform "${CMAKE_SYSTEM_NAME}" --json
  OUTPUT_VARIABLE _tvm_tirx_config OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(JSON TVM_TIRX_VERSION GET "${_tvm_tirx_config}" version)
string(JSON tvm_tirx_INCLUDE_DIR GET "${_tvm_tirx_config}" include_dir)
string(JSON _tvm_tirx_openmp GET "${_tvm_tirx_config}" openmp)
string(JSON _tvm_tirx_custom_logging GET "${_tvm_tirx_config}" custom_logging)
string(JSON _tvm_tirx_use_threads GET "${_tvm_tirx_config}" use_threads)
if(_tvm_tirx_use_threads)
  find_dependency(Threads)
endif()
set(tvm_tirx_VERSION "${TVM_TIRX_VERSION}")
set(TVM_TIRX_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(TVM_TIRX_BUILD_FOR_ANDROID "${ANDROID}")

# FFI owns its headers, library, and CMake helpers independently of TIRx.
if(NOT TARGET tvm_ffi::shared)
  if(NOT tvm_ffi_DIR)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
      OUTPUT_VARIABLE tvm_ffi_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
  endif()
  find_dependency(tvm_ffi CONFIG)
endif()

# ELF RUNPATH is not transitive. Retain a direct FFI dependency while preserving
# the consumer's linker state. A group keeps the flags attached if the consumer
# also links FFI explicitly and CMake deduplicates or reorders that dependency.
set(_tvm_tirx_ffi_link tvm_ffi::shared)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_LINK_GROUP_USING_tvm_tirx_ffi_needed
    "LINKER:--push-state,--no-as-needed" "LINKER:--pop-state")
  set(CMAKE_LINK_GROUP_USING_tvm_tirx_ffi_needed_SUPPORTED TRUE)
  set(_tvm_tirx_ffi_link "$<LINK_GROUP:tvm_tirx_ffi_needed,tvm_ffi::shared>")
endif()

if(NOT TARGET tvm_tirx::header)
  add_library(tvm_tirx::header INTERFACE IMPORTED)
  target_compile_features(tvm_tirx::header INTERFACE cxx_std_20)
  target_compile_definitions(tvm_tirx::header INTERFACE TVM_INDEX_DEFAULT_I64=1)
  if(_tvm_tirx_custom_logging)
    target_compile_definitions(tvm_tirx::header INTERFACE TVM_LOG_CUSTOMIZE=1)
  endif()
  target_include_directories(tvm_tirx::header INTERFACE "${tvm_tirx_INCLUDE_DIR}")
  target_link_libraries(tvm_tirx::header INTERFACE tvm_ffi::header)
endif()

foreach(_library tvm_runtime tvm_tirx tvm_runtime_static
    tvm_runtime_cuda tvm_runtime_opencl tvm_runtime_vulkan tvm_runtime_metal tvm_runtime_rocm)
  string(JSON _location ERROR_VARIABLE _missing
    GET "${_tvm_tirx_config}" libraries ${_library} location)
  if(_missing OR TARGET tvm_tirx::${_library})
    continue()
  endif()
  if(_library STREQUAL "tvm_runtime_static")
    add_library(tvm_tirx::${_library} STATIC IMPORTED)
  else()
    add_library(tvm_tirx::${_library} SHARED IMPORTED)
  endif()
  set_target_properties(tvm_tirx::${_library} PROPERTIES IMPORTED_LOCATION "${_location}")
  if(WIN32 AND NOT _library STREQUAL "tvm_runtime_static")
    string(JSON _implib GET "${_tvm_tirx_config}" libraries ${_library} implib)
    set_target_properties(tvm_tirx::${_library} PROPERTIES IMPORTED_IMPLIB "${_implib}")
  endif()
  target_link_libraries(tvm_tirx::${_library} INTERFACE tvm_tirx::header)
  if(_library STREQUAL "tvm_runtime" OR _library STREQUAL "tvm_runtime_static")
    target_link_libraries(tvm_tirx::${_library} INTERFACE "${_tvm_tirx_ffi_link}")
    if(_tvm_tirx_use_threads)
      target_link_libraries(tvm_tirx::${_library} INTERFACE Threads::Threads)
    endif()
  else()
    target_link_libraries(tvm_tirx::${_library} INTERFACE tvm_tirx::tvm_runtime)
  endif()
endforeach()

if(TARGET tvm_tirx::tvm_runtime_static)
  target_link_libraries(tvm_tirx::tvm_runtime_static INTERFACE ${CMAKE_DL_LIBS})
  if(WIN32)
    # Only the runtime is static: declarations for the separate FFI DLL stay intact.
    target_compile_definitions(tvm_tirx::tvm_runtime_static INTERFACE
      TVM_RUNTIME_DLL= TVM_RUNTIME_DLL_EXPORT=)
  endif()
  if(_tvm_tirx_openmp STREQUAL "gnu")
    find_dependency(OpenMP COMPONENTS CXX)
    target_link_libraries(tvm_tirx::tvm_runtime_static INTERFACE OpenMP::OpenMP_CXX)
  elseif(_tvm_tirx_openmp STREQUAL "intel")
    find_library(_tvm_tirx_iomp NAMES iomp5 libiomp5md REQUIRED)
    target_link_libraries(tvm_tirx::tvm_runtime_static INTERFACE "${_tvm_tirx_iomp}")
  endif()
endif()

if(TARGET tvm_tirx::tvm_runtime_opencl)
  if(WIN32)
    target_compile_definitions(tvm_tirx::tvm_runtime_opencl INTERFACE
      "TVM_OPENCL_DLL=__declspec(dllimport)")
  else()
    target_compile_definitions(tvm_tirx::tvm_runtime_opencl INTERFACE
      "TVM_OPENCL_DLL=__attribute__((visibility(\"default\")))")
  endif()
endif()
set(TVM_TIRX_USE_CUDA OFF)
set(TVM_TIRX_USE_ROCM OFF)
if(TARGET tvm_tirx::tvm_runtime_cuda)
  set(TVM_TIRX_USE_CUDA ON)
endif()
if(TARGET tvm_tirx::tvm_runtime_rocm)
  set(TVM_TIRX_USE_ROCM ON)
endif()
