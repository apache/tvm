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

function(generate_crt_config platform output_path)
  set(TVM_CRT_DEBUG 0)
  set(TVM_CRT_MAX_NDIM 6)
  set(TVM_CRT_MAX_ARGS 10)
  set(TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES 512)
  set(TVM_CRT_MAX_REGISTERED_MODULES 2)
  set(TVM_CRT_MAX_PACKET_SIZE_BYTES 2048)
  set(TVM_CRT_MAX_STRLEN_DLTYPE 10)
  set(TVM_CRT_MAX_STRLEN_FUNCTION_NAME 120)
  set(TVM_CRT_MAX_STRLEN_PARAM_NAME 80)

  if("${platform}" STREQUAL "zephyr")
    set(TVM_CRT_MAX_PACKET_SIZE_BYTES 512)
  elseif("${platform}" STREQUAL "arduino")
    set(TVM_CRT_MAX_PACKET_SIZE_BYTES 8*1024)
  endif()
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/crt/crt_config.h.template" "${output_path}")
endfunction()
