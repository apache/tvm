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

# On MacOS, the default C compiler (/usr/bin/cc) is actually a small script that dispatches to a
# compiler the default SDK (usually /Library/Developer/CommandLineTools/usr/bin/ or
# /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/). CMake
# automatically detects what is being dispatched and uses it instead along with all the flags it
# needs. CMake makes this second compiler available through the CMAKE_C_COMPILER variable, but it
# does not make the necessary flags available. This leads to configuration errors in libbacktrace
# because it can't find system libraries. Our solution is to detect if CMAKE_C_COMPILER lives in
# /Library or /Applications and switch to the default compiler instead.
include(ExternalProject)


if(CMAKE_SYSTEM_NAME MATCHES "Darwin" AND (CMAKE_C_COMPILER MATCHES "^/Library"
  OR CMAKE_C_COMPILER MATCHES "^/Applications"))
    set(c_compiler "/usr/bin/cc")
  else()
    set(c_compiler "${CMAKE_C_COMPILER}")
endif()

ExternalProject_Add(project_libbacktrace
  PREFIX libbacktrace
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/libbacktrace
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace
  CONFIGURE_COMMAND "${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/libbacktrace/configure"
                    "--prefix=${CMAKE_CURRENT_BINARY_DIR}/libbacktrace"
                    --with-pic
                    "CC=${c_compiler}"
                    "CFLAGS=${CMAKE_C_FLAGS}"
                    "LDFLAGS=${CMAKE_EXE_LINKER_FLAGS}"
                    "CPP=${c_compiler} -E"
                    "NM=${CMAKE_NM}"
                    "STRIP=${CMAKE_STRIP}"
                    "--host=${MACHINE_NAME}"
  INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace"
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/lib/libbacktrace.a"
                   "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include/backtrace.h"
  )

# Custom step to rebuild libbacktrace if any of the source files change
tvm_file_glob(GLOB LIBBACKTRACE_SRCS "${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/libbacktrace/*.c")
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
