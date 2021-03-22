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

ExternalProject_Add(project_libbacktrace
  PREFIX libbacktrace
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/libbacktrace
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace
  CONFIGURE_COMMAND "${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/libbacktrace/configure"
                    "--prefix=${CMAKE_CURRENT_BINARY_DIR}/libbacktrace" --with-pic
  INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace"
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/lib/libbacktrace.a"
                   "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include/backtrace.h"
  )

# Custom step to rebuild libbacktrace if any of the source files change
file(GLOB LIBBACKTRACE_SRCS "${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/libbacktrace/*.c")
ExternalProject_Add_Step(project_libbacktrace checkout
  DEPENDERS configure
  DEPENDEES download
  DEPENDS ${LIBBACKTRACE_SRCS}
)

add_library(libbacktrace STATIC IMPORTED)
add_dependencies(libbacktrace project_libbacktrace)
set_property(TARGET libbacktrace
  PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/lib/libbacktrace.a)
# create include directory so cmake doesn't complain
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include)
