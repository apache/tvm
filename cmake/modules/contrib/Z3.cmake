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

# src/arith/z3_prover.cc is always part of COMPILER_SRCS (picked up by the
# src/arith/*.cc glob). It compiles a conservative stub by default and switches
# to the real Z3 implementation only when the TVM_USE_Z3 macro is defined below.
if(NOT USE_Z3)
  return()
endif()

find_package(Z3 QUIET)
set(Z3_PYTHON_RESULT 1)

if(NOT Z3_FOUND)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_EXECUTABLE)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import z3; print(z3.__path__[0])"
      OUTPUT_VARIABLE Z3_PYTHON_PACKAGE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE Z3_PYTHON_RESULT
    )
  endif()

  if(Z3_PYTHON_RESULT EQUAL 0 AND NOT Z3_PYTHON_PACKAGE_DIR STREQUAL "")
    find_path(Z3_INCLUDE_DIR NO_DEFAULT_PATH NAMES z3++.h PATHS "${Z3_PYTHON_PACKAGE_DIR}/include")
    find_library(
      Z3_LIBRARY
      NO_DEFAULT_PATH
      NAMES z3 libz3
      PATHS "${Z3_PYTHON_PACKAGE_DIR}/bin" "${Z3_PYTHON_PACKAGE_DIR}/lib"
            "${Z3_PYTHON_PACKAGE_DIR}/lib64"
    )
  endif()
endif()

if(TARGET z3::libz3 OR TARGET Z3::libz3)
  if(TARGET z3::libz3)
    set(Z3_TARGET z3::libz3)
  else()
    set(Z3_TARGET Z3::libz3)
  endif()
  get_target_property(Z3_TARGET_INCLUDE_DIRS ${Z3_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  if(Z3_TARGET_INCLUDE_DIRS)
    include_directories(SYSTEM ${Z3_TARGET_INCLUDE_DIRS})
  endif()
  list(APPEND TVM_LINKER_LIBS ${Z3_TARGET})
elseif(Z3_FOUND OR (Z3_INCLUDE_DIR AND Z3_LIBRARY))
  if(NOT Z3_INCLUDE_DIR AND Z3_CXX_INCLUDE_DIRS)
    set(Z3_INCLUDE_DIR ${Z3_CXX_INCLUDE_DIRS})
  endif()
  if(NOT Z3_LIBRARY AND Z3_LIBRARIES)
    set(Z3_LIBRARY ${Z3_LIBRARIES})
  endif()
  if(NOT Z3_INCLUDE_DIR OR NOT Z3_LIBRARY)
    message(FATAL_ERROR "USE_Z3 is ON, but Z3 include directory or library was not found.")
  endif()
  include_directories(SYSTEM ${Z3_INCLUDE_DIR})
  list(APPEND TVM_LINKER_LIBS ${Z3_LIBRARY})
else()
  message(FATAL_ERROR "USE_Z3 is ON, but Z3 was not found. Install Z3 or PyPI z3-solver.")
endif()

# Enable the real Z3 implementation inside the single src/arith/z3_prover.cc file.
add_compile_definitions(TVM_USE_Z3)
message(STATUS "Build with Z3 SMT solver support")
