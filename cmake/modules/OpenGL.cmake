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

find_package(OpenGL QUIET)

if(OpenGL_FOUND)
  # always set the includedir when dir is available
  # avoid global retrigger of cmake
  include_directories(${OPENGL_INCLUDE_DIRS})
endif(OpenGL_FOUND)

if(USE_OPENGL)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 QUIET REQUIRED)
  message(STATUS "Build with OpenGL support")
  file(GLOB RUNTIME_OPENGL_SRCS src/runtime/opengl/*.cc)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${OpenGL_LIBRARIES} glfw)
  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENGL_SRCS})
else(USE_OPENGL)
  list(APPEND COMPILER_SRCS src/codegen/opt/build_opengl_off.cc)
endif(USE_OPENGL)
