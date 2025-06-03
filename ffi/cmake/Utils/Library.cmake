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
function(add_dsymutil target_name)
  # running dsymutil on macos to generate debugging symbols for backtraces
  if(APPLE AND TVM_FFI_USE_LIBBACKTRACE)
    find_program(DSYMUTIL dsymutil)
    mark_as_advanced(DSYMUTIL)
    add_custom_command(TARGET ${target_name}
        POST_BUILD
        COMMAND ${DSYMUTIL} ARGS $<TARGET_FILE:${target_name}>
        COMMENT "[COMMAND] dsymutil $<TARGET_FILE:${target_name}>"
        VERBATIM
    )
  endif()
endfunction()

function(add_msvc_flags target_name)
  # running if we are under msvc
  if(MSVC)
    target_compile_definitions(${target_name} PUBLIC -DWIN32_LEAN_AND_MEAN)
    target_compile_definitions(${target_name} PUBLIC -D_CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(${target_name} PUBLIC -D_SCL_SECURE_NO_WARNINGS)
    target_compile_definitions(${target_name} PUBLIC -D_ENABLE_EXTENDED_ALIGNED_STORAGE)
    target_compile_definitions(${target_name} PUBLIC -DNOMINMAX)
    target_compile_options(${target_name} PRIVATE "/Z7")
  endif()
endfunction()

function(add_target_from_obj target_name obj_target_name)
  add_library(${target_name}_static STATIC $<TARGET_OBJECTS:${obj_target_name}>)
  set_target_properties(
    ${target_name}_static PROPERTIES
    OUTPUT_NAME "${target_name}_static"
    PREFIX "lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
  add_library(${target_name}_shared SHARED $<TARGET_OBJECTS:${obj_target_name}>)
  set_target_properties(
    ${target_name}_shared PROPERTIES
    OUTPUT_NAME "${target_name}"
    PREFIX "lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  add_custom_target(${target_name})
  add_dependencies(${target_name} ${target_name}_static ${target_name}_shared)
  if (MSVC)
    target_compile_definitions(${obj_target_name} PRIVATE TVM_FFI_EXPORTS)
  endif()
  add_dsymutil(${target_name}_shared)
  add_msvc_flags(${target_name}_shared)
endfunction()
