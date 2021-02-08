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
include(ExternalProject)

# Git tag for libbacktrace
set(TAG dedbe13fda00253fe5d4f2fb812c909729ed5937)

ExternalProject_Add(project_libbacktrace
  PREFIX libbacktrace
  GIT_REPOSITORY https://github.com/ianlancetaylor/libbacktrace.git
  GIT_TAG "${TAG}"
  CONFIGURE_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/src/project_libbacktrace/configure"
                    "--prefix=${CMAKE_CURRENT_BINARY_DIR}/libbacktrace" --with-pic
  INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace"
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/lib/libbacktrace.a"
                   "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include/backtrace.h"
  # disable the builtin update because it rebuilds on every build
  UPDATE_COMMAND ""
  )

# Only rebuild libbacktrace if this file changes
# libbacktrace has a bug on macOS with shared libraries: when looking for symbols in shared
# libraries, if any library is missing symbols, then the process stops and no symbols are used. We
# include a patch that changes libbacktrace to instead perform a best-effort lookup of symbols. This
# is a one-line change where a return is changed to a continue within the symbol lookup loop.
ExternalProject_Add_Step(project_libbacktrace checkout
  DEPENDERS configure
  DEPENDEES download
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/Libbacktrace.cmake" "${CMAKE_CURRENT_LIST_DIR}/libbacktrace_macos.patch"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/src/project_libbacktrace
  COMMAND git checkout -f ${TAG};
  COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/libbacktrace_macos.patch
  COMMENT "update and patch"
  )

add_library(libbacktrace STATIC IMPORTED)
add_dependencies(libbacktrace project_libbacktrace)
set_property(TARGET libbacktrace
  PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/lib/libbacktrace.a)
# create include directory so cmake doesn't complain
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include)
