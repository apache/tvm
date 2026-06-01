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

include(FetchContent)

# Function: _libxml2compile — fetch libxml2 (no submodule) and produce a
# static LibXml2 target with all optional deps disabled. Used when
# USE_STATIC_LIBXML2_FROM_SOURCE=ON to avoid depending on a system libxml2.
function(_libxml2compile)
  FetchContent_Declare(
    libxml2
    GIT_REPOSITORY https://github.com/GNOME/libxml2.git
    GIT_TAG        v2.15.0
  )

  # Static build, all optional features OFF.
  # Anything not listed here keeps upstream default (typically cascade-OFF
  # via libxml2's own cmake_dependent_option logic). See task file for the
  # full cross-check.
  set(BUILD_SHARED_LIBS        OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_PROGRAMS    OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_TESTS       OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_PYTHON      OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_DOCS        OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_ICU         OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_ICONV       OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_LEGACY      OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_ZLIB        OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_READLINE    OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_MODULES     OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_HTTP        OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_HTML        OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_CATALOG     OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_DEBUG       OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_XINCLUDE    OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_XPATH       OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_VALID       OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_REGEXPS     OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_PATTERN     OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_SAX1        OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_OUTPUT      OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_PUSH        OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_ISO8859X    OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_TLS         OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_THREAD_ALLOC OFF CACHE INTERNAL "")
  set(LIBXML2_WITH_THREADS     OFF CACHE INTERNAL "")  # TVM never calls into libxml2

  FetchContent_MakeAvailable(libxml2)

  # Windows requires LIBXML_STATIC defined on consumers of the static lib.
  if(WIN32)
    target_compile_definitions(LibXml2 INTERFACE LIBXML_STATIC)
  endif()
endfunction()
