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
find_cuda(${USE_CUDA} ${USE_CUDNN})

if(CUDA_FOUND)
  # always set the includedir when CUDA is available
  # avoid global retrigger of CMake
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

if(USE_CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
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
  # Furthermore, this override cannot be specified in a user's
  # `config.cmake` for TVM, because it must be set after CMake's
  # built-in CUDA support.
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
endif(USE_CUDA)

if(USE_CUDA)
  message(STATUS "Build cuda device runtime")

  # tvm_runtime_cuda links libcuda; without it the .so silently drops its
  # libcuda.so.1 dependency, so fail configure instead.
  if(NOT CUDA_CUDA_LIBRARY)
    message(FATAL_ERROR "USE_CUDA is on but libcuda was not found. "
      "Set -DCUDA_CUDA_LIBRARY=<path to libcuda.so> or make the driver stub discoverable.")
  endif()

  tvm_file_glob(GLOB RUNTIME_CUDA_SRCS src/backend/cuda/runtime/*.cc)
  tvm_file_glob(GLOB VM_CUDA_BUILTIN_SRC_CC src/runtime/vm/cuda/*.cc)

  add_library(tvm_runtime_cuda_objs OBJECT ${RUNTIME_CUDA_SRCS} ${VM_CUDA_BUILTIN_SRC_CC})
  target_link_libraries(tvm_runtime_cuda_objs PUBLIC tvm_ffi_header)
  # These sources compile into tvm_runtime_cuda.dll, so their TVM_RUNTIME_DLL /
  # TVM_FFI_DLL symbols must be dllexport on MSVC (e.g. GetCudaDeviceCount in
  # cuda_device_api.cc). Mirror tvm_runtime_objs; a no-op on non-MSVC platforms.
  target_compile_definitions(tvm_runtime_cuda_objs PRIVATE TVM_RUNTIME_EXPORTS TVM_FFI_EXPORTS)
  set_target_properties(tvm_runtime_cuda_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(TVM_VISIBILITY_FLAG)
    target_compile_options(tvm_runtime_cuda_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
  endif()
  add_library(tvm_runtime_cuda SHARED $<TARGET_OBJECTS:tvm_runtime_cuda_objs>)
  list(APPEND TVM_RUNTIME_BACKEND_LIBS tvm_runtime_cuda)
  target_link_libraries(tvm_runtime_cuda PUBLIC tvm_runtime ${CUDA_CUDART_LIBRARY} ${CUDA_CUDA_LIBRARY})
  tvm_configure_target_library(tvm_runtime_cuda RUNTIME_MODULE)

  if(USE_NVTX)
    message(STATUS "Build with NVTX support")
    target_link_libraries(tvm_runtime_cuda PRIVATE ${CUDA_NVTX_LIBRARY})
  endif()
endif(USE_CUDA)

# Contrib sources gated by USE_CUDA go into libtvm_runtime_extra.
# See the RuntimeExtra assembly block in CMakeLists.txt.

if(USE_CUDA AND USE_CUDNN)
  message(STATUS "Build with cuDNN support")
  include_directories(SYSTEM ${CUDA_CUDNN_INCLUDE_DIRS})
  tvm_file_glob(GLOB CUDNN_RELAX_CONTRIB_SRC src/relax/backend/contrib/cudnn/*.cc)
  list(APPEND COMPILER_SRCS ${CUDNN_RELAX_CONTRIB_SRC})
  tvm_file_glob(GLOB CONTRIB_CUDNN_SRCS src/runtime/extra/contrib/cudnn/*.cc)
  add_library(tvm_cudnn_objs OBJECT ${CONTRIB_CUDNN_SRCS})
  target_link_libraries(tvm_cudnn_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_cudnn_objs ${CUDA_CUDNN_LIBRARY})
endif(USE_CUDA AND USE_CUDNN)

if(USE_CUDA AND USE_CUDNN_FRONTEND)
  message(STATUS "Build with cuDNN Frontend support")
  if(IS_DIRECTORY ${USE_CUDNN_FRONTEND})
    find_file(CUDNN_FRONTEND_HEADER cudnn_frontend.h HINTS ${USE_CUDNN_FRONTEND}/include)
    include_directories(SYSTEM ${USE_CUDNN_FRONTEND}/include)
  else()
    find_file(CUDNN_FRONTEND_HEADER cudnn_frontend.h)
  endif()
  if(NOT CUDNN_FRONTEND_HEADER)
    message(FATAL_ERROR "Cannot find cudnn_frontend.h, please set USE_CUDNN_FRONTEND to the path of the cuDNN frontend header")
  endif()
  tvm_file_glob(GLOB CONTRIB_CUDNN_FRONTEND_SRCS src/runtime/extra/contrib/cudnn/cudnn_frontend/*.cc)
  set_source_files_properties(${CONTRIB_CUDNN_SRCS} PROPERTIES COMPILE_DEFINITIONS TVM_USE_CUDNN_FRONTEND=1)
  add_library(tvm_cudnn_frontend_objs OBJECT ${CONTRIB_CUDNN_FRONTEND_SRCS})
  target_link_libraries(tvm_cudnn_frontend_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_cudnn_frontend_objs)
endif(USE_CUDA AND USE_CUDNN_FRONTEND)

if(USE_CUDA AND USE_CUBLAS)
  message(STATUS "Build with cuBLAS support")
  tvm_file_glob(GLOB CUBLAS_CONTRIB_SRC src/relax/backend/contrib/cublas/*.cc)
  list(APPEND COMPILER_SRCS ${CUBLAS_CONTRIB_SRC})
  tvm_file_glob(GLOB CONTRIB_CUBLAS_SRCS src/runtime/extra/contrib/cublas/*.cc)
  add_library(tvm_cublas_objs OBJECT ${CONTRIB_CUBLAS_SRCS})
  target_link_libraries(tvm_cublas_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_cublas_objs ${CUDA_CUBLAS_LIBRARY})
  if(NOT CUDA_CUBLASLT_LIBRARY STREQUAL "CUDA_CUBLASLT_LIBRARY-NOTFOUND")
    target_link_libraries(tvm_runtime_extra PRIVATE ${CUDA_CUBLASLT_LIBRARY})
  endif()
endif(USE_CUDA AND USE_CUBLAS)

if(USE_CUDA AND USE_THRUST)
  message(STATUS "Build with Thrust support")
  tvm_file_glob(GLOB CONTRIB_THRUST_SRC src/runtime/extra/contrib/thrust/*.cu)
  add_library(tvm_thrust_objs OBJECT ${CONTRIB_THRUST_SRC})
  target_link_libraries(tvm_thrust_objs PRIVATE tvm_runtime_extra_defs)
  target_compile_options(tvm_thrust_objs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  if(NOT USE_THRUST MATCHES ${IS_TRUE_PATTERN})
    find_package(CCCL REQUIRED COMPONENTS Thrust)
    target_link_libraries(tvm_thrust_objs PRIVATE CCCL::Thrust)
  endif()
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_thrust_objs)
endif(USE_CUDA AND USE_THRUST)

if(USE_CUDA AND USE_CURAND)
  message(STATUS "Build with cuRAND support")
  message(STATUS "${CUDA_CURAND_LIBRARY}")
  tvm_file_glob(GLOB CONTRIB_CURAND_SRC_CC src/runtime/extra/contrib/curand/*.cc)
  tvm_file_glob(GLOB CONTRIB_CURAND_SRC_CU src/runtime/extra/contrib/curand/*.cu)
  add_library(tvm_curand_objs OBJECT ${CONTRIB_CURAND_SRC_CC} ${CONTRIB_CURAND_SRC_CU})
  target_link_libraries(tvm_curand_objs PRIVATE tvm_runtime_extra_defs)
  target_link_libraries(tvm_runtime_extra PRIVATE tvm_curand_objs ${CUDA_CURAND_LIBRARY})
endif(USE_CUDA AND USE_CURAND)
