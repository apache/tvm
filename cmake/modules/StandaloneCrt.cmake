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

if(USE_STANDALONE_CRT)
  message(STATUS "Build with standalone CRT")
  file(GLOB crt_srcs src/runtime/crt/**)

  function(tvm_crt_add_copy_file var src dest)
    get_filename_component(basename "${src}" NAME)
    get_filename_component(dest_parent_dir "${dest}" DIRECTORY)
    add_custom_command(
        OUTPUT "${dest}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${src}" "${dest}"
        DEPENDS "${src}")
    list(APPEND "${var}" "${dest}")
    set("${var}" "${${var}}" PARENT_SCOPE)
  endfunction(tvm_crt_add_copy_file)

  # Build an isolated build directory, separate from the TVM tree.
  file(GLOB_RECURSE crt_srcs
       RELATIVE "${CMAKE_SOURCE_DIR}/src/runtime/crt"
       "${CMAKE_SOURCE_DIR}/src/runtime/crt/common/*.c"
       "${CMAKE_SOURCE_DIR}/src/runtime/crt/graph_runtime/*.c"
       "${CMAKE_SOURCE_DIR}/src/runtime/crt/include/*.h")

  foreach(src IN LISTS crt_srcs)
    tvm_crt_add_copy_file(host_isolated_build_deps ${CMAKE_SOURCE_DIR}/src/runtime/crt/${src} standalone_crt/${src})
  endforeach()

  file(GLOB_RECURSE crt_headers RELATIVE "${CMAKE_SOURCE_DIR}/include" include/tvm/runtime/crt/*.h)
  foreach(hdr IN LISTS crt_headers)
    tvm_crt_add_copy_file(host_isolated_build_deps ${CMAKE_SOURCE_DIR}/include/${hdr} standalone_crt/include/${hdr})
  endforeach()

  tvm_crt_add_copy_file(host_isolated_build_deps
      ${CMAKE_SOURCE_DIR}/include/tvm/runtime/c_runtime_api.h standalone_crt/include/tvm/runtime/c_runtime_api.h)
  tvm_crt_add_copy_file(host_isolated_build_deps
      ${CMAKE_SOURCE_DIR}/include/tvm/runtime/c_backend_api.h standalone_crt/include/tvm/runtime/c_backend_api.h)
  tvm_crt_add_copy_file(host_isolated_build_deps
      ${CMAKE_SOURCE_DIR}/src/runtime/crt/Makefile standalone_crt/Makefile)

  get_filename_component(crt_config_abspath src/runtime/crt/host/crt_config.h ABSOLUTE)
  list(APPEND host_isolated_build_deps src/runtime/crt/host/crt_config.h)

  get_filename_component(host_build_dir_abspath "${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt" ABSOLUTE)

  if(${VERBOSE})
  set(make_quiet QUIET=)
  else(${VERBOSE})
  set(make_quiet )
  endif(${VERBOSE})
  add_custom_command(
      OUTPUT host_standalone_crt/common/libcommon.a host_standalone_crt/graph_runtime/libgraph_runtime.a
      COMMAND make
          DLPACK_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/3rdparty/dlpack/include
          TVM_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/standalone_crt/include
          CRT_CONFIG=${crt_config_abspath}
          BUILD_DIR=${host_build_dir_abspath} all ${make_quiet}
      WORKING_DIRECTORY standalone_crt
      DEPENDS ${host_isolated_build_deps})
  add_custom_target(host_standalone_crt SOURCES  host_standalone_crt/common/libcommon.a host_standalone_crt/graph_runtime/libgraph_runtime.a)

#  add_custom_target(host_standalone_crt ALL
#      DEPENDS host_standalone_crt/common/libcommon.a host_standalone_crt/graph_runtime/libgraph_runtime.a)
  add_library(host_standalone_crt_common INTERFACE)
  add_dependencies(host_standalone_crt_common host_standalone_crt)
  target_link_libraries(host_standalone_crt_common
      INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/crt/common/libcommon.a)
  # set_target_properties(host_standalone_crt_common PROPERTIES
  #     IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/crt/common
  #     IMPORTED_OBJECTS ${CMAKE_CURRENT_BINARY_DIR}/crt/common/libcommon.a
  #     PUBLIC_HEADER ${crt_headers})
  add_dependencies(host_standalone_crt_common host_standalone_crt)
#      ${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt/common/libcommon.a)

  add_library(host_standalone_crt_graph_runtime INTERFACE IMPORTED)
  add_dependencies(host_standalone_crt_graph_runtime host_standalone_crt)
  target_link_libraries(host_standalone_crt_graph_runtime
      INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/crt/graph_runtime/libgraph_runtime.a)

  # set_target_properties(host_standalone_crt_graph_runtime PROPERTIES
  #     IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/crt/graph_runtime
  #     IMPORTED_OBJECTS ${CMAKE_CURRENT_BINARY_DIR}/crt/graph_runtime/libgraph_runtime.a
  #     PUBLIC_HEADER ${crt_headers})
  add_dependencies(host_standalone_crt_graph_runtime host_standalone_crt)
#      ${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt/graph_runtime/libgraph_runtime.a)

  # Standalone CRT tests
  file(GLOB TEST_SRCS ${CMAKE_SOURCE_DIR}/tests/crt/*.cc)
  find_path(GTEST_INCLUDE_DIR gtest/gtest.h)
  find_library(GTEST_LIB gtest "$ENV{GTEST_LIB}")

  # Create the `crttest` target if we can find GTest.  If not, we create dummy
  # targets that give the user an informative error message.
  if(GTEST_INCLUDE_DIR AND GTEST_LIB)
    foreach(__srcpath ${TEST_SRCS})
      get_filename_component(__srcname ${__srcpath} NAME)
      string(REPLACE ".cc" "" __execname ${__srcname})
      add_executable(${__execname} ${__srcpath})
      list(APPEND TEST_EXECS ${__execname})
      target_include_directories(${__execname} PUBLIC ${GTEST_INCLUDE_DIR})
      target_link_directories(${__execname} PRIVATE
          ${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt/common
          ${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt/graph_runtime)
      target_link_libraries(${__execname} common graph_runtime ${GTEST_LIB})
      set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_ALL 1)
      set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
    endforeach()
    add_custom_target(crttest DEPENDS ${TEST_EXECS})
  elseif(NOT GTEST_INCLUDE_DIR)
    add_custom_target(crttest
        COMMAND echo "Missing Google Test headers in include path"
        COMMAND exit 1)
  elseif(NOT GTEST_LIB)
    add_custom_target(crttest
        COMMAND echo "Missing Google Test library"
        COMMAND exit 1)
  endif()

endif(USE_STANDALONE_CRT)
