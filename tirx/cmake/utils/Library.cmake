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

# Apply compilation policy only to TIRx targets, including their object libraries.
function(tvm_tirx_configure_target target)
  set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
  set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
  set_property(TARGET ${target} PROPERTY CUDA_STANDARD 20)
  set_property(TARGET ${target} PROPERTY CUDA_STANDARD_REQUIRED ON)
  set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)
  set_property(TARGET ${target} PROPERTY CXX_VISIBILITY_PRESET hidden)
  set_property(TARGET ${target} PROPERTY VISIBILITY_INLINES_HIDDEN ON)

  # Keep shared libraries relocatable within their installation directory.
  if(APPLE)
    set_target_properties(${target} PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
  elseif(UNIX)
    set_target_properties(${target} PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
  endif()
endfunction()
