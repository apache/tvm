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
# tvm_set_relative_rpath(target_name)
#
# Give a target a relative rpath ($ORIGIN / @loader_path) so that sibling
# shared libraries in the same directory resolve each other regardless of the
# install location (e.g. inside a Python wheel). No-op if the target does not
# exist.
function(tvm_set_relative_rpath target_name)
  if(NOT TARGET ${target_name})
    return()
  endif()

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
endfunction()

#######################################################
# tvm_configure_runtime_module(target_name)
#
# Apply the standard layout for an optional runtime backend library
# (tvm_runtime_cuda, tvm_runtime_vulkan, ...): emit it into the build lib
# directory, give it a wheel-relative rpath, and install it. When building the
# Python module it is additionally installed into the package "lib" directory.
# No-op if the target does not exist.
#
# Parameters:
#   target_name: CMake target to configure
function(tvm_configure_runtime_module target_name)
  if(NOT TARGET ${target_name})
    return()
  endif()

  set_target_properties(${target_name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  tvm_set_relative_rpath(${target_name})
  install(TARGETS ${target_name} DESTINATION lib${LIB_SUFFIX})
  if(TVM_BUILD_PYTHON_MODULE)
    install(TARGETS ${target_name} DESTINATION "lib")
  endif()
endfunction()
