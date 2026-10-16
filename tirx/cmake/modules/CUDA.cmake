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

# CUDA Module
find_cuda(${TVM_TIRX_USE_CUDA} OFF)

if(CUDA_FOUND)
  # always set the includedir when CUDA is available
  # avoid global retrigger of CMake
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

if(TVM_TIRX_USE_CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA, TVM_TIRX_USE_CUDA=" ${TVM_TIRX_USE_CUDA})
  endif()
  message(STATUS "Build with CUDA ${CUDA_VERSION} support")
  enable_language(CUDA)

  # Ensure that include directives to NVCC are in the
  # `compile_commands.json`, as required by clangd.
  #
  # As of CMake 3.29.5 [0], if the NVCC version is 11 or higher, CMake
  # will generate a "options-file.rsp" containing the -I flags for
  # include directories, rather than providing them on the
  # command-line.  This setting exists to work around the short
  # command-line length limits on Windows, but is enabled on all
  # platforms.  If set, because include directories are not part of
  # the `compile_commands.json`, the clangd LSP cannot find the
  # include files.
  #
  # Set this after enabling CMake's built-in CUDA support.
  #
  # [0] https://github.com/Kitware/CMake/commit/6377a438
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_LESS "3.24")
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES not set. Please upgrade CMake to 3.24 to use native, or set CMAKE_CUDA_ARCHITECTURES manually")
    endif()
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, using native")
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
endif(TVM_TIRX_USE_CUDA)

if(TVM_TIRX_USE_CUDA)
  message(STATUS "Build cuda device runtime")

  # tvm_runtime_cuda links libcuda; without it the .so silently drops its
  # libcuda.so.1 dependency, so fail configure instead.
  if(NOT CUDA_CUDA_LIBRARY)
    message(FATAL_ERROR "TVM_TIRX_USE_CUDA is on but libcuda was not found. "
      "Set -DCUDA_CUDA_LIBRARY=<path to libcuda.so> or make the driver stub discoverable.")
  endif()

  file(GLOB RUNTIME_CUDA_SRCS CONFIGURE_DEPENDS src/backend/cuda/runtime/*.cc)

  add_library(tvm_runtime_cuda_objs OBJECT ${RUNTIME_CUDA_SRCS})
  target_link_libraries(tvm_runtime_cuda_objs PUBLIC tvm_tirx_header)
  # These sources compile into tvm_runtime_cuda.dll, so their TVM_RUNTIME_DLL /
  # TVM_FFI_DLL symbols must be dllexport on MSVC (e.g. GetCudaDeviceCount in
  # cuda_device_api.cc). Mirror tvm_runtime_objs; a no-op on non-MSVC platforms.
  target_compile_definitions(tvm_runtime_cuda_objs PRIVATE TVM_RUNTIME_EXPORTS TVM_FFI_EXPORTS)
  set_target_properties(tvm_runtime_cuda_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(tvm_runtime_cuda SHARED $<TARGET_OBJECTS:tvm_runtime_cuda_objs>)
  install(TARGETS tvm_runtime_cuda DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm_tirx)
  list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_cuda)
  target_link_libraries(tvm_runtime_cuda PUBLIC tvm_runtime PRIVATE ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY})
  tvm_tirx_configure_target(tvm_runtime_cuda_objs)
  tvm_tirx_configure_target(tvm_runtime_cuda)

endif(TVM_TIRX_USE_CUDA)
