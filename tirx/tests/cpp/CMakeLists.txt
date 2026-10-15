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

# C++ tests always use GTest. Prefer an existing package, otherwise build it.
set(TVM_TIRX_USE_OPENCL_GTEST "" CACHE PATH "Optional OpenCL-specific GTest installation")
if(TVM_TIRX_USE_OPENCL_GTEST)
  set(GTest_ROOT "${TVM_TIRX_USE_OPENCL_GTEST}")
endif()
find_package(GTest CONFIG QUIET)
if(TARGET GTest::gtest AND TARGET GTest::gtest_main AND TARGET GTest::gmock)
  set(_gtest_targets GTest::gtest GTest::gtest_main GTest::gmock)
else()
  include(FetchContent)
  set(INSTALL_GTEST OFF)
  set(BUILD_GMOCK ON)
  set(gtest_force_shared_crt ON)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
  FetchContent_MakeAvailable(googletest)
  # Use one coherent fetched build even when a partial installed package exists.
  set(_gtest_targets gtest gtest_main gmock)
endif()
include(GoogleTest)
file(GLOB_RECURSE _test_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
add_executable(tirx_cpptest ${_test_sources})
target_link_libraries(tirx_cpptest PRIVATE tvm_tirx ${_gtest_targets} ${CMAKE_DL_LIBS})
if(TVM_TIRX_USE_THREADS)
  target_link_libraries(tirx_cpptest PRIVATE Threads::Threads)
endif()
# LLVM symbols are hidden in the compiler library, including static LLVM builds.
if(DEFINED LLVM_LIBS)
  target_link_libraries(tirx_cpptest PRIVATE ${LLVM_LIBS})
endif()
set_target_properties(tirx_cpptest PROPERTIES
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
target_compile_definitions(tirx_cpptest PRIVATE NDEBUG)
if(APPLE)
  set_property(TARGET tirx_cpptest PROPERTY BUILD_RPATH "@loader_path/lib")
elseif(UNIX)
  set_property(TARGET tirx_cpptest PROPERTY BUILD_RPATH "\$ORIGIN/lib")
endif()
gtest_discover_tests(tirx_cpptest WORKING_DIRECTORY "${PROJECT_BINARY_DIR}" DISCOVERY_MODE PRE_TEST)
