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
#  NCCL_ROOT - When set, this path is inspected instead of standard library
#              locations as the root of the NCCL installation.
#              The environment variable NCCL_ROOT overrides this variable.
#
# This module defines
#  Nccl_FOUND, whether nccl has been found
#  NCCL_INCLUDE_DIR, directory containing header
#  NCCL_LIBRARY, directory containing nccl library
# This module assumes that the user has already called find_package(CUDA)

macro(find_nccl use_nccl)
  if(${use_nccl} MATCHES ${IS_FALSE_PATTERN})
    return()
  endif()
  set(NCCL_LIB_NAME nccl_static)
  if(${use_nccl} MATCHES ${IS_TRUE_PATTERN})
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h)
    find_library(NCCL_LIBRARY NAMES nccl_static)
  else()
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS ${use_nccl} ${use_nccl}/include)
    find_library(NCCL_LIBRARY NAMES nccl_static HINTS ${use_nccl} ${use_nccl}/lib)
  endif()
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Nccl DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)
  if (Nccl_FOUND)
    message(STATUS "Found NCCL_LIBRARY: ${NCCL_LIBRARY}")
    message(STATUS "Found NCCL_INCLUDE_DIR: ${NCCL_INCLUDE_DIR}")
    add_library(nccl SHARED IMPORTED)
    set_target_properties(nccl
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
      IMPORTED_LOCATION             "${NCCL_LIBRARY}")
  else()
    message(STATUS "NCCL not found")
  endif()
  mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
endmacro(find_nccl)
