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
# This script provides:
# - zstd_FOUND
# - zstd_INCLUDE_DIR
# - zstd_LIBRARY
# - zstd_STATIC_LIBRARY
#
# And also defines the following targets:
# - zstd::libzstd_static
# - zstd::libzstd_shared (a dummy one to work around LLVM checking)
if(MSVC)
  set(zstd_STATIC_LIBRARY_SUFFIX "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
else()
  set(zstd_STATIC_LIBRARY_SUFFIX "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
endif()

find_path(zstd_INCLUDE_DIR NAMES zstd.h)
find_library(zstd_LIBRARY        NAMES zstd zstd_static)
find_library(zstd_STATIC_LIBRARY NAMES      zstd_static
  "${CMAKE_STATIC_LIBRARY_PREFIX}zstd${CMAKE_STATIC_LIBRARY_SUFFIX}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(zstd DEFAULT_MSG zstd_LIBRARY zstd_INCLUDE_DIR)

if(zstd_FOUND)
  message(STATUS "Found zstd_STATIC_LIBRARY: ${zstd_STATIC_LIBRARY}")
  message(STATUS "Found zstd_INCLUDE_DIR: ${zstd_INCLUDE_DIR}")
  if(zstd_STATIC_LIBRARY MATCHES "${zstd_STATIC_LIBRARY_SUFFIX}$" AND
     NOT TARGET zstd::libzstd_static)
    add_library(zstd::libzstd_static STATIC IMPORTED)
    set_target_properties(zstd::libzstd_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${zstd_INCLUDE_DIR}"
        IMPORTED_LOCATION "${zstd_STATIC_LIBRARY}")
    add_library(zstd::libzstd_shared SHARED IMPORTED)
  endif()
endif()

unset(zstd_STATIC_LIBRARY_SUFFIX)
mark_as_advanced(zstd_INCLUDE_DIR zstd_LIBRARY zstd_STATIC_LIBRARY)
