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

if(USE_VITIS_AI)
  set(PYXIR_SHARED_LIB libpyxir.so)
  find_package(PythonInterp 3.7 REQUIRED)
  if(NOT PYTHON)
    find_program(PYTHON NAMES python3 python3.8)
  endif()
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import pyxir as px; print(px.get_include_dir()); print(px.get_lib_dir());"
    RESULT_VARIABLE __result
    OUTPUT_VARIABLE __output
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(__result MATCHES 0)
    string(REGEX REPLACE ";" "\\\\;" __values ${__output})
    string(REGEX REPLACE "\r?\n" ";"    __values ${__values})
    list(GET __values 0 PYXIR_INCLUDE_DIR)
    list(GET __values 1 PYXIR_LIB_DIR)
  else()
    message(FATAL_ERROR "Can't build TVM with Vitis-AI because PyXIR can't be found")
  endif()
  message(STATUS "Build with contrib.vitisai")
  include_directories(${PYXIR_INCLUDE_DIR})
  tvm_file_glob(GLOB VAI_CONTRIB_SRC src/runtime/contrib/vitis_ai/*.cc)
  tvm_file_glob(GLOB COMPILER_VITIS_AI_SRCS
                src/relay/backend/contrib/vitis_ai/*)
  list(APPEND COMPILER_SRCS ${COMPILER_VITIS_AI_SRCS})
  link_directories(${PYXIR_LIB_DIR})
  list(APPEND TVM_RUNTIME_LINKER_LIBS "pyxir")
  list(APPEND RUNTIME_SRCS ${VAI_CONTRIB_SRC})
endif(USE_VITIS_AI)
