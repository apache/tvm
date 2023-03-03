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
# Enhanced version of find CSI-NN2 compute library.
#
# Usage:
#   find_csinn(${use_csinn}, ${target})
#
# - When use_csinn=ON, use auto search
# - When use_csinn=/path/to/csinn2-library-path, use the path.
# - WHEN environment variable CSINN_PATH is set, use that path
#   Can be useful when cross compiling and cannot rely on
#   CMake to provide the correct library as part of the
#   requested toolchain.
# - When target=X86, use csinn2 dynamic library for x86
# - When target=C906, use csinn2 dynamic library for rv C906
# Provide variables:
#
# - CSINN_FOUND
# - CSINN_INCLUDE_DIRS
# - CSINN_LIBRARIES
#
macro(find_csinn use_csinn target)
  set(__use_csinn ${use_csinn})
  set(__target ${target})
  if(IS_DIRECTORY ${__use_csinn})
    set(__csinn_library ${__use_csinn})
    message(STATUS "Custom CSI-NN2 compute library PATH=" ${__use_csinn})
  elseif(IS_DIRECTORY $ENV{CSINN_PATH})
    set(__csinn_library $ENV{CSINN_PATH})
  else()
    set(__csinn_library "")
  endif()
  if(__csinn_library)
    set(CSINN_INCLUDE_DIRS ${__csinn_library}/include)
    if (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY STREQUAL "ONLY")
      set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    endif()
    if(__target STREQUAL "X86")
      find_library(CSINN_LIBRARIES NAMES libcsi_nn2_ref_x86.so PATHS ${__csinn_library}/install/lib)
    elseif(__target STREQUAL "C906")
      find_library(CSINN_LIBRARIES NAMES libcsi_nn2_c906.so PATHS ${__csinn_library}/install/lib)
    else()
      find_library(CSINN_LIBRARIES NAMES libcsi_nn2_ref_x86.so PATHS ${__csinn_library}/install/lib)
    endif()
    if(CSINN_LIBRARIES)
      set(CSINN_FOUND TRUE)
    endif()
  endif(__csinn_library)
  # No user provided CSINN include/libs found
  if(NOT CSINN_FOUND)
    if(${__use_csinn} MATCHES ${IS_TRUE_PATTERN})
      find_package(CSI-NN2 QUIET)
    endif()
  endif()
  if(CSINN_FOUND)
    message(STATUS "CSINN_INCLUDE_DIRS=" ${CSINN_INCLUDE_DIRS})
    message(STATUS "CSINN_LIBRARIES=" ${CSINN_LIBRARIES})
  endif(CSINN_FOUND)
endmacro(find_csinn)