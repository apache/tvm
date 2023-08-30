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
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
# This module defines
#  Rccl_FOUND, whether rccl has been found
#  RCCL_INCLUDE_DIR, directory containing header
#  RCCL_LIBRARY, directory containing rccl library
# This module assumes that the user has already called find_package(rocm)

macro(find_rccl use_rccl)
  if(${use_rccl} MATCHES ${IS_FALSE_PATTERN})
    return()
  endif()
  if(${use_rccl} MATCHES ${IS_TRUE_PATTERN})
    find_path(RCCL_INCLUDE_DIR NAMES rccl.h)
    find_library(RCCL_LIBRARY NAMES rccl)
  else()
    find_path(RCCL_INCLUDE_DIR NAMES rccl.h HINTS ${use_rccl} ${use_rccl}/include)
    find_library(RCCL_LIBRARY NAMES rccl HINTS ${use_rccl} ${use_rccl}/lib)
  endif()
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Rccl DEFAULT_MSG RCCL_INCLUDE_DIR RCCL_LIBRARY)
  if (Rccl_FOUND)
    message(STATUS "Found RCCL_LIBRARY: ${RCCL_LIBRARY}")
    message(STATUS "Found RCCL_INCLUDE_DIR: ${RCCL_INCLUDE_DIR}")
    add_library(rccl SHARED IMPORTED)
    set_target_properties(rccl
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${RCCL_INCLUDE_DIR}"
      IMPORTED_LOCATION             "${RCCL_LIBRARY}")
  else()
    message(STATUS "RCCL not found")
  endif()
  mark_as_advanced(RCCL_INCLUDE_DIR RCCL_LIBRARY)
endmacro(find_rccl)
