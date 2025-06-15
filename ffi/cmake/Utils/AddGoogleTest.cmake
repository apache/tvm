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
set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_GetProperties(googletest)
if (NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  message(STATUS "Found googletest_SOURCE_DIR - ${googletest_SOURCE_DIR}")
  message(STATUS "Found googletest_BINARY_DIR - ${googletest_BINARY_DIR}")
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  include(GoogleTest)
  set_target_properties(gtest      PROPERTIES EXPORT_COMPILE_COMMANDS OFF EXCLUDE_FROM_ALL ON FOLDER 3rdparty)
  set_target_properties(gtest_main PROPERTIES EXPORT_COMPILE_COMMANDS OFF EXCLUDE_FROM_ALL ON FOLDER 3rdparty)
  set_target_properties(gmock      PROPERTIES EXPORT_COMPILE_COMMANDS OFF EXCLUDE_FROM_ALL ON FOLDER 3rdparty)
  set_target_properties(gmock_main PROPERTIES EXPORT_COMPILE_COMMANDS OFF EXCLUDE_FROM_ALL ON FOLDER 3rdparty)
  mark_as_advanced(
      BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
      gmock_build_tests gtest_build_samples gtest_build_tests
      gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  )
endif()

macro(add_googletest target_name)
  add_test(
    NAME ${target_name}
    COMMAND ${target_name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(${target_name} PRIVATE gtest_main)
  gtest_discover_tests(${target_name}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DISCOVERY_MODE PRE_TEST
    PROPERTIES
      VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
  set_target_properties(${target_name} PROPERTIES FOLDER tests)
endmacro()
