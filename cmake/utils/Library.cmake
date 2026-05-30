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
# tvm_set_python_module_relative_rpath(target_name)
#
# Give a target a relative rpath ($ORIGIN / @loader_path) so that, inside a
# Python wheel, intra-package shared libraries resolve each other from their
# own directory. No-op unless TVM_BUILD_PYTHON_MODULE is set and the target
# exists.
function(tvm_set_python_module_relative_rpath target_name)
  if(NOT TVM_BUILD_PYTHON_MODULE OR NOT TARGET ${target_name})
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
