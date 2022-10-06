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

# This script configures the logging module and dependency on libbacktrace

if(USE_CUSTOM_LOGGING)
  # Set and propogate TVM_LOG_CUSTOMIZE flag is custom logging has been requested
  target_compile_definitions(tvm_objs PUBLIC TVM_LOG_CUSTOMIZE=1)
  target_compile_definitions(tvm_runtime_objs PUBLIC TVM_LOG_CUSTOMIZE=1)
  target_compile_definitions(tvm_libinfo_objs PUBLIC TVM_LOG_CUSTOMIZE=1)
  target_compile_definitions(tvm PUBLIC TVM_LOG_CUSTOMIZE=1)
  target_compile_definitions(tvm_runtime PUBLIC TVM_LOG_CUSTOMIZE=1)
endif()

if("${USE_LIBBACKTRACE}" STREQUAL "AUTO")
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(USE_LIBBACKTRACE ON)
  else()
    set(USE_LIBBACKTRACE OFF)
  endif()
  message(STATUS "Autoset: USE_LIBBACKTRACE=" ${USE_LIBBACKTRACE} " in " ${CMAKE_SYSTEM_NAME})
endif()


if(USE_LIBBACKTRACE)
  message(STATUS "Building with libbacktrace...")
  include(cmake/libs/Libbacktrace.cmake)
  target_link_libraries(tvm PRIVATE libbacktrace)
  target_link_libraries(tvm_runtime PRIVATE libbacktrace)
  add_dependencies(tvm_runtime_objs libbacktrace)
  # pre 3.12 versions of cmake cannot propagate include directories from imported targets so we set them manually
  target_include_directories(tvm PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include")
  target_include_directories(tvm_objs PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include")
  target_include_directories(tvm_runtime PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include")
  target_include_directories(tvm_runtime_objs PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include")
  target_compile_definitions(tvm_objs PRIVATE TVM_USE_LIBBACKTRACE=1)
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_USE_LIBBACKTRACE=1)
else()
  target_compile_definitions(tvm_objs PRIVATE TVM_USE_LIBBACKTRACE=0)
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_USE_LIBBACKTRACE=0)
endif()

if(BACKTRACE_ON_SEGFAULT)
  target_compile_definitions(tvm_objs PRIVATE TVM_BACKTRACE_ON_SEGFAULT)
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_BACKTRACE_ON_SEGFAULT)
endif()
