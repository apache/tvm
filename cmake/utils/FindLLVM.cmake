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

#######################################################
# Enhanced version of find llvm.
#
# Usage:
#   find_llvm(${USE_LLVM} [${LLVM_VERSION}])
#
# - When USE_LLVM=ON, use auto search
# - When USE_LLVM=/path/to/llvm-config, use corresponding config
# - The optional LLVM_VERSION=<required version> which will find a matching version
#      If LLVM_VERSION=10 then it will find 10.X.X, any version of 10
#      If LLVM_VERSION=10.2 then it will find 10.2.X
#
# Provide variables:
# - LLVM_INCLUDE_DIRS
# - LLVM_LIBS
# - LLVM_DEFINITIONS
# - TVM_LLVM_VERSION
# - TVM_INFO_LLVM_VERSION
#
macro(find_llvm use_llvm)
  if(${use_llvm} MATCHES ${IS_FALSE_PATTERN})
    return()
  endif()
  set(LLVM_CONFIG ${use_llvm})
  if(${ARGC} EQUAL 2)
    set(llvm_version_required ${ARGV1})
  endif()

  if(${LLVM_CONFIG} MATCHES ${IS_TRUE_PATTERN})
    find_package(LLVM ${llvm_version_required} REQUIRED CONFIG)
    llvm_map_components_to_libnames(LLVM_LIBS "all")
    if (NOT LLVM_LIBS)
      message(STATUS "Not found - LLVM_LIBS")
      message(STATUS "Fall back to using llvm-config")
      set(LLVM_CONFIG "${LLVM_TOOLS_BINARY_DIR}/llvm-config")
    endif()
  endif()

  if(LLVM_LIBS) # check if defined, not if it is true
    list (FIND LLVM_LIBS "LLVM" _llvm_dynlib_index)
    if (${_llvm_dynlib_index} GREATER -1)
      set(LLVM_LIBS LLVM)
      message(STATUS "Link with dynamic LLVM library")
    else()
      list(REMOVE_ITEM LLVM_LIBS LTO)
      message(STATUS "Link with static LLVM libraries")
    endif()
    set(TVM_LLVM_VERSION ${LLVM_VERSION_MAJOR}${LLVM_VERSION_MINOR})
    set(TVM_INFO_LLVM_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
  else()
    # use llvm config
    message(STATUS "Use llvm-config=" ${LLVM_CONFIG})
    separate_arguments(LLVM_CONFIG)
    execute_process(COMMAND ${LLVM_CONFIG} --libfiles
      RESULT_VARIABLE __llvm_exit_code
      OUTPUT_VARIABLE __llvm_libfiles_space
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT "${__llvm_exit_code}" STREQUAL "0")
      message(FATAL_ERROR "Fatal error executing: ${LLVM_CONFIG} --libfiles")
    endif()
    execute_process(COMMAND ${LLVM_CONFIG} --system-libs
      RESULT_VARIABLE __llvm_exit_code
      OUTPUT_VARIABLE __llvm_system_libs
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT "${__llvm_exit_code}" STREQUAL "0")
      message(FATAL_ERROR "Fatal error executing: ${LLVM_CONFIG} --system-libs")
    endif()
    execute_process(COMMAND ${LLVM_CONFIG} --cxxflags
      RESULT_VARIABLE __llvm_exit_code
      OUTPUT_VARIABLE __llvm_cxxflags_space
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT "${__llvm_exit_code}" STREQUAL "0")
      message(FATAL_ERROR "Fatal error executing: ${LLVM_CONFIG} --cxxflags")
    endif()
    execute_process(COMMAND ${LLVM_CONFIG} --version
      RESULT_VARIABLE __llvm_exit_code
      OUTPUT_VARIABLE __llvm_version
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT "${__llvm_exit_code}" STREQUAL "0")
      message(FATAL_ERROR "Fatal error executing: ${LLVM_CONFIG} --version")
    endif()
    execute_process(COMMAND ${LLVM_CONFIG} --prefix
      RESULT_VARIABLE __llvm_exit_code
      OUTPUT_VARIABLE __llvm_prefix
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT "${__llvm_exit_code}" STREQUAL "0")
      message(FATAL_ERROR "Fatal error executing: ${LLVM_CONFIG} --prefix")
    endif()
    execute_process(COMMAND ${LLVM_CONFIG} --libdir
      RESULT_VARIABLE __llvm_exit_code
      OUTPUT_VARIABLE __llvm_libdir
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT "${__llvm_exit_code}" STREQUAL "0")
      message(FATAL_ERROR "Fatal error executing: ${LLVM_CONFIG} --libdir")
    endif()
    message(STATUS "LLVM libdir: ${__llvm_libdir}")
    execute_process(COMMAND ${LLVM_CONFIG} --cmakedir
      RESULT_VARIABLE __llvm_exit_code
      OUTPUT_VARIABLE __llvm_cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT "${__llvm_exit_code}" STREQUAL "0")
      message(FATAL_ERROR "Fatal error executing: ${LLVM_CONFIG} --cmakedir")
    endif()
    cmake_path(SET "__llvm_cmakedir" "${__llvm_cmakedir}")
    message(STATUS "LLVM cmakedir: ${__llvm_cmakedir}")
    # map prefix => $
    # to handle the case when the prefix contains space.
    string(REPLACE ${__llvm_prefix} "$" __llvm_cxxflags ${__llvm_cxxflags_space})
    string(REPLACE ${__llvm_prefix} "$" __llvm_libfiles ${__llvm_libfiles_space})
    # llvm version
    set(TVM_INFO_LLVM_VERSION ${__llvm_version})
    string(REGEX REPLACE "^([^.]+)\.([^.])+\.[^.]+.*$" "\\1\\2" TVM_LLVM_VERSION ${__llvm_version})
    string(STRIP ${TVM_LLVM_VERSION} TVM_LLVM_VERSION)
    # definitions
    string(REGEX MATCHALL "(^| )-D[A-Za-z0-9_]*" __llvm_defs ${__llvm_cxxflags})
    set(LLVM_DEFINITIONS "")
    foreach(__flag IN ITEMS ${__llvm_defs})
      string(STRIP "${__flag}" __llvm_def)
      list(APPEND LLVM_DEFINITIONS "${__llvm_def}")
    endforeach()
    # include dir
    string(REGEX MATCHALL "(^| )-I[^ ]*" __llvm_include_flags ${__llvm_cxxflags})
    set(LLVM_INCLUDE_DIRS "")
    foreach(__flag IN ITEMS ${__llvm_include_flags})
      string(REGEX REPLACE "(^| )-I" "" __dir "${__flag}")
      # map $ => prefix
      string(REPLACE "$" ${__llvm_prefix} __dir_with_prefix "${__dir}")
      list(APPEND LLVM_INCLUDE_DIRS "${__dir_with_prefix}")
    endforeach()
    # libfiles
    set(LLVM_LIBS "")
    separate_arguments(__llvm_libfiles)
    foreach(__flag IN ITEMS ${__llvm_libfiles})
      # map $ => prefix
      string(REPLACE "$" ${__llvm_prefix} __lib_with_prefix "${__flag}")
      list(APPEND LLVM_LIBS "${__lib_with_prefix}")
    endforeach()
    if (${USE_MLIR})
      if (EXISTS "${__llvm_libdir}/libMLIRPresburger.a")
        if (EXISTS "${__llvm_libdir}/libMLIRSupport.a")
          message(STATUS "Found MLIR")
          list(APPEND LLVM_LIBS "${__llvm_libdir}/libMLIRPresburger.a")
          list(APPEND LLVM_LIBS "${__llvm_libdir}/libMLIRSupport.a")
          set(TVM_MLIR_VERSION ${TVM_LLVM_VERSION})
          message(STATUS "Build with MLIR")
          message(STATUS "Set TVM_MLIR_VERSION=" ${TVM_MLIR_VERSION})
        endif()
      endif()
    endif()
    separate_arguments(__llvm_system_libs)
    foreach(__flag IN ITEMS ${__llvm_system_libs})
      if("${__flag}" STREQUAL "-lm")
        message(STATUS "LLVM links against math")
        list(APPEND LLVM_LIBS "m")
      elseif(("${__flag}" STREQUAL "-lz") OR ("${__flag}" STREQUAL "z.lib"))
        message(STATUS "LLVM links against zlib")
        find_package(ZLIB REQUIRED)
        list(APPEND LLVM_LIBS "ZLIB::ZLIB")
      elseif("${__flag}" STREQUAL "-lzstd" OR ("${__flag}" STREQUAL "zstd.dll.lib"))
        list(APPEND CMAKE_MODULE_PATH "${__llvm_cmakedir}")
        find_package(zstd REQUIRED)
        if (TARGET "zstd::libzstd_static")
          message(STATUS "LLVM links against static zstd")
          list(APPEND LLVM_LIBS "zstd::libzstd_static")
        else()
          message(STATUS "LLVM links against shared zstd")
          list(APPEND LLVM_LIBS "zstd::libzstd_shared")
        endif()
      elseif("${__flag}" STREQUAL "-lxml2")
        message(STATUS "LLVM links against xml2")
        list(APPEND LLVM_LIBS "-lxml2")
      elseif((__flag MATCHES ".lib$") AND (EXISTS "${__llvm_libdir}/${__flag}"))
        # If the library file ends in .lib try to also search the llvm_libdir
        message(STATUS "LLVM linker flag under LLVM libdir: ${__llvm_libdir}/${__flag}")
        list(APPEND LLVM_LIBS "${__llvm_libdir}/${__flag}")
      else()
        message(STATUS "LLVM linker flag: ${__flag}")
        list(APPEND LLVM_LIBS "${__flag}")
      endif()
    endforeach()
  endif()
  message(STATUS "Found LLVM_INCLUDE_DIRS=" "${LLVM_INCLUDE_DIRS}")
  message(STATUS "Found LLVM_DEFINITIONS=" "${LLVM_DEFINITIONS}")
  message(STATUS "Found LLVM_LIBS=" "${LLVM_LIBS}")
  message(STATUS "Found TVM_LLVM_VERSION=" ${TVM_LLVM_VERSION})
  if (${TVM_LLVM_VERSION} LESS 40)
    message(FATAL_ERROR "TVM requires LLVM 4.0 or higher.")
  endif()
endmacro(find_llvm)
