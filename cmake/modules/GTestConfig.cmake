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

set(Build_GTests OFF)
if(NOT TARGET gtest)
  unset(runtime_gtests)
  if(DEFINED USE_OPENCL_GTEST AND EXISTS ${USE_OPENCL_GTEST})
    set(runtime_gtests ${USE_OPENCL_GTEST})
  elseif(DEFINED USE_VULKAN_GTEST AND EXISTS ${USE_VULKAN_GTEST})
    set(runtime_gtests ${USE_VULKAN_GTEST})
  elseif(ANDROID_ABI AND DEFINED ENV{ANDROID_NDK_HOME})
    set(GOOGLETEST_ROOT $ENV{ANDROID_NDK_HOME}/sources/third_party/googletest)
    add_library(gtest_main STATIC
      ${GOOGLETEST_ROOT}/src/gtest_main.cc
      ${GOOGLETEST_ROOT}/src/gtest-all.cc)
    target_include_directories(gtest_main PRIVATE ${GOOGLETEST_ROOT})
    target_include_directories(gtest_main PUBLIC ${GOOGLETEST_ROOT}/include)
    set(Build_GTests ON)
    message(STATUS "Using gtest from Android NDK")
    return()
  else()
    message(STATUS "No valid GTest path found, skipping GTest configuration")
    return()
  endif()

  # Configure if runtime_gtests is valid
  if(runtime_gtests AND EXISTS ${runtime_gtests})
    include(FetchContent)
    FetchContent_Declare(googletest SOURCE_DIR "${runtime_gtests}")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    install(TARGETS gtest EXPORT ${PROJECT_NAME}Targets DESTINATION lib${LIB_SUFFIX})
    set(Build_GTests ON)
  else()
    set(Build_GTests OFF)
    return()
  endif()
endif()
