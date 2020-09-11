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
# Find Arm Ethos-N libraries
#
# Usage:
#   find_ethosn(${USE_ETHOSN})
#
# - When USE_ETHOSN=/path/to/ethos-sdk-path, use the path from USE_ETHOSN
# - Else, when environment variable ETHOSN_STACK is set, use that path
# - When USE_ETHOSN=ON, use auto search
#
# Provide variables:
#
# - ETHOSN_FOUND
# - ETHOSN_PACKAGE_VERSION
# - ETHOSN_DEFINITIONS
# - ETHOSN_INCLUDE_DIRS
# - ETHOSN_COMPILER_LIBRARY
# - ETHOSN_RUNTIME_LIBRARY

macro(find_ethosn use_ethosn)
  set(__use_ethosn ${use_ethosn})
  if(IS_DIRECTORY ${__use_ethosn})
    set(__ethosn_stack ${__use_ethosn})
    message(STATUS "Arm Ethos-N driver stack PATH=" ${__use_ethosn})
  elseif(IS_DIRECTORY $ENV{ETHOSN_STACK})
     set(__ethosn_stack $ENV{ETHOSN_STACK})
    message(STATUS "Arm Ethos-N driver stack from env=" ${__use_ethosn})
  else()
     set(__ethosn_stack "")
  endif()

  if(__ethosn_stack)
    set(ETHOSN_INCLUDE_DIRS "")
    # Compile-time support
    find_path(_SL_DIR NAMES Support.hpp
      PATHS ${__ethosn_stack}/include/ethosn_support_library)
    string(REGEX REPLACE "/ethosn_support_library" "" _SL_DIR2 ${_SL_DIR})
    list(APPEND ETHOSN_INCLUDE_DIRS "${_SL_DIR2}")

    find_library(ETHOSN_COMPILER_LIBRARY NAMES EthosNSupport
      PATHS ${__ethosn_stack}/lib)
    find_library(ETHOSN_COMPILER_LIBRARY NAMES EthosNSupport)

    set(ETHOSN_PACKAGE_VERSION "0.1.1")

    if(USE_ETHOSN_HW STREQUAL "ON")
      # Runtime hardware support
      find_path(_DL_DIR NAMES Network.hpp
        PATHS ${__ethosn_stack}/include/ethosn_driver_library)
      string(REGEX REPLACE "/ethosn_driver_library" "" _DL_DIR2 ${_DL_DIR})
      list(APPEND ETHOSN_INCLUDE_DIRS "${_DL_DIR2}")

      find_library(ETHOSN_RUNTIME_LIBRARY NAMES EthosNDriver
        PATHS ${__ethosn_stack}/lib)
      find_library(ETHOSN_RUNTIME_LIBRARY NAMES EthosNDriver)
      set(ETHOSN_DEFINITIONS -DETHOSN_HW)
    endif ()

    if(ETHOSN_COMPILER_LIBRARY)
      set(ETHOSN_FOUND TRUE)
    endif()
  endif(__ethosn_stack)

  if(NOT ETHOSN_FOUND)
    if(__use_ethosn STREQUAL "ON")
      message(WARNING "No cmake find_package available for Arm Ethos-N")
    endif()

  # additional libraries
  else()
    message(STATUS "Found ETHOSN_DEFINITIONS=${ETHOSN_DEFINITIONS}")
    message(STATUS "Found ETHOSN_INCLUDE_DIRS=${ETHOSN_INCLUDE_DIRS}")
    message(STATUS "Found ETHOSN_COMPILER_LIBRARY=${ETHOSN_COMPILER_LIBRARY}")
    message(STATUS "Found ETHOSN_RUNTIME_LIBRARY=${ETHOSN_RUNTIME_LIBRARY}")
  endif(NOT ETHOSN_FOUND)

endmacro(find_ethosn)
