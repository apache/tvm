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

# Helpers for configuring library targets.

#######################################################
# tvm_configure_target_library(target_name [RUNTIME_MODULE])
#
# Configure a TVM library target. The target always gets a relative rpath
# ($ORIGIN / @loader_path) so that sibling shared libraries in the same
# directory resolve each other regardless of the install location (e.g. inside a
# Python wheel).
#
# With the RUNTIME_MODULE option -- used for the optional runtime backend
# libraries (tvm_runtime_cuda, tvm_runtime_vulkan, ...) -- the target is also
# emitted into the build "lib" directory and installed; when building the Python
# module it is additionally installed into the package "lib" directory. Targets
# that manage their own output directory / install rules (tvm_compiler,
# tvm_runtime, ...) omit the option and take only the rpath.
#
# No-op if the target does not exist.
function(tvm_configure_target_library target_name)
  if(NOT TARGET ${target_name})
    return()
  endif()
  cmake_parse_arguments(ARG "RUNTIME_MODULE" "" "" ${ARGN})

  if(APPLE)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "@loader_path"
      INSTALL_RPATH "@loader_path"
    )
  elseif(UNIX)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "\$ORIGIN"
      INSTALL_RPATH "\$ORIGIN"
    )
  endif()

  if(ARG_RUNTIME_MODULE)
    set_target_properties(${target_name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
    install(TARGETS ${target_name} DESTINATION lib${LIB_SUFFIX})
    if(TVM_BUILD_PYTHON_MODULE)
      install(TARGETS ${target_name} DESTINATION "lib")
    endif()
  endif()
endfunction()
