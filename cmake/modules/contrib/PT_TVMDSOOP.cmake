# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(NOT USE_PT_TVMDSOOP STREQUAL "OFF")
  find_package(PythonInterp REQUIRED)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.__path__[0].strip())"
    OUTPUT_VARIABLE PT_PATH
    RESULT_VARIABLE PT_STATUS)

  if(NOT ${PT_STATUS} EQUAL 0)
    message(FATAL_ERROR "Fail to get pytorch path")
  endif()

  string(REGEX REPLACE "\n" "" PT_PATH "${PT_PATH}")
  message(STATUS "PyTorch path: ${PT_PATH}")

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch;print(torch.compiled_with_cxx11_abi())"
    OUTPUT_VARIABLE PT_CXX_FLAG
    RESULT_VARIABLE PT_STATUS)

  string(REGEX REPLACE "\n" "" PT_CXX_FLAG "${PT_CXX_FLAG}")
  message(STATUS "Found TORCH_BUILT_WITH_CXX_ABI=${PT_CXX_FLAG} ")

  if(${PT_CXX_FLAG} STREQUAL "False")
    set(CXX_ABI_ENABLED 0)
  else()
    set(CXX_ABI_ENABLED 1)
  endif()

  set_property(
    SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/contrib/torch/tvm_module_wrapper/RuntimeModuleWrapperTorch.cc
    APPEND PROPERTY
    COMPILE_OPTIONS
    "-D_GLIBCXX_USE_CXX11_ABI=${CXX_ABI_ENABLED}"
    "-I${PT_PATH}/include"
  )

  set_property(
    SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/contrib/torch/pt_call_tvm/tvm_class.cc
    APPEND PROPERTY
    COMPILE_OPTIONS
    "-I${PT_PATH}/include"
  )

  set(PT_LINK_FLAGS_STR "-L${PT_PATH}/lib -l:libtorch.so -l:libtorch_python.so")

  if(NOT USE_CUDA STREQUAL "OFF")
    add_definitions(-DPT_TVMDSOOP_ENABLE_GPU)
  endif()

  string(REGEX REPLACE "\n" " " PT_FLAGS "${PT_COMPILE_FLAGS} ${PT_LINK_FLAGS}")
  separate_arguments(PT_COMPILE_FLAGS UNIX_COMMAND)
  separate_arguments(PT_LINK_FLAGS UNIX_COMMAND ${PT_LINK_FLAGS_STR})

  # This old version is depereated and will be removed after tvm 0.11
  set(LIBRARY_OLD_NAME pt_tvmdsoop)

  # This new library is set for pytorch integration, which solves the c++ abi imcompability issue
  set(LIBRARY_NEW_NAME pt_tvmdsoop_new)
  tvm_file_glob(GLOB_RECURSE PTTVM_TORCH ${CMAKE_CURRENT_SOURCE_DIR}/src/contrib/torch/tvm_module_wrapper/*.cc)

  tvm_file_glob(GLOB_RECURSE PTTVM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/contrib/torch/pt_call_tvm/*.cc)

  add_library(${LIBRARY_OLD_NAME} SHARED ${PTTVM_SRCS})
  add_library(${LIBRARY_NEW_NAME} SHARED ${PTTVM_TORCH})
  set(PTTVM_LINK_FLAGS -ltvm -L${CMAKE_CURRENT_BINARY_DIR})

  if(NOT BUILD_PT_TVMDSOOP_ONLY STREQUAL "ON")
    add_dependencies(${LIBRARY_OLD_NAME} tvm)
    add_dependencies(${LIBRARY_NEW_NAME} tvm)
  endif()

  target_compile_options(${LIBRARY_OLD_NAME} PUBLIC ${PTTVM_COMPILE_FLAGS} ${PT_COMPILE_FLAGS})
  target_link_libraries(${LIBRARY_OLD_NAME} PUBLIC ${PTTVM_LINK_FLAGS} ${PT_LINK_FLAGS})
  target_compile_definitions(${LIBRARY_OLD_NAME} PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

  target_compile_options(${LIBRARY_NEW_NAME} PUBLIC ${PTTVM_COMPILE_FLAGS} ${PT_COMPILE_FLAGS})
  target_link_libraries(${LIBRARY_NEW_NAME} PUBLIC ${PTTVM_LINK_FLAGS} ${PT_LINK_FLAGS})
  target_compile_definitions(${LIBRARY_NEW_NAME} PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
endif()
