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
  include(ExternalProject)

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
  set(CRC16_PATH "${CMAKE_SOURCE_DIR}/3rdparty/mbed-os/targets/TARGET_NORDIC/TARGET_NRF5x/TARGET_SDK_11/libraries/crc16")
  list(APPEND CRT_FILE_COPY_JOBS
       "3rdparty/mbed-os/targets/TARGET_NORDIC/TARGET_NRF5x/TARGET_SDK_11/libraries/crc16 *.h -> include *.c -> src/runtime/crt/utvm_rpc_server"
       "3rdparty/dlpack/include *.h -> include"
       "3rdparty/dmlc-core/include *.h -> include"
       "include/tvm/runtime c_*_api.h -> include/tvm/runtime"
       "include/tvm/runtime/crt *.h -> include/tvm/runtime/crt"
       "src/runtime/crt Makefile -> ."
       "src/runtime/crt/include *.h -> include"
       "src/runtime/crt/common *.c -> src/runtime/crt/common"
       "src/runtime/crt/graph_runtime *.c -> src/runtime/crt/graph_runtime"
       "src/runtime/crt/utvm_rpc_server * -> src/runtime/crt/utvm_rpc_server"
       "src/runtime/crt/host crt_config.h -> src/runtime/crt/host"
       "src/runtime/minrpc * -> src/runtime/minrpc"
       "src/support generic_arena.h -> src/support"
       )

  foreach(job_spec IN LISTS CRT_FILE_COPY_JOBS)
    string(REPLACE " " ";" job_spec "${job_spec}")
    list(LENGTH job_spec job_spec_length)
    math(EXPR job_spec_length_mod "${job_spec_length} % 3")
    if(NOT "${job_spec_length_mod}" EQUAL 1)
      message(FATAL_ERROR "CRT copy job spec list length is ${job_spec_length}; parsed job spec is ${job_spec}")
    endif()
    math(EXPR job_spec_stop "${job_spec_length} - 3")

    list(GET job_spec 0 job_src_base)
    set(job_src_base "${CMAKE_SOURCE_DIR}/${job_src_base}")
    foreach(copy_pattern_index RANGE 1 "${job_spec_stop}" 3)
      list(GET job_spec ${copy_pattern_index} copy_pattern)
      math(EXPR copy_dest_index "${copy_pattern_index} + 2")
      list(GET job_spec ${copy_dest_index} copy_dest)

      file(GLOB_RECURSE copy_files
           RELATIVE "${job_src_base}"
           "${job_src_base}/${copy_pattern}")
      list(LENGTH copy_files copy_files_length)
      if("${copy_files_length}" EQUAL 0)
        message(FATAL_ERROR "CRT copy job matched 0 files: ${job_src_base}/${copy_pattern} -> ${copy_dest}")
      endif()
      foreach(copy_src IN LISTS copy_files)
        tvm_crt_add_copy_file(host_isolated_build_deps ${job_src_base}/${copy_src} standalone_crt/${copy_dest}/${copy_src})
      endforeach()
    endforeach()
  endforeach()

#  get_filename_component(crt_config_abspath src/runtime/crt/host/crt_config.h ABSOLUTE)
#  list(APPEND host_isolated_build_deps src/runtime/crt/host/crt_config.h)
  add_custom_target(standalone_crt DEPENDS ${host_isolated_build_deps})

  get_filename_component(host_build_dir_abspath "${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt" ABSOLUTE)

  if(${VERBOSE})
  set(make_quiet QUIET=)
  else(${VERBOSE})
  set(make_quiet )
  endif(${VERBOSE})

  list(APPEND crt_libraries common graph_runtime utvm_rpc_server)
  list(TRANSFORM crt_libraries PREPEND "host_standalone_crt/lib" OUTPUT_VARIABLE crt_library_paths)
  list(TRANSFORM crt_library_paths APPEND ".a" OUTPUT_VARIABLE crt_library_paths)
  ExternalProject_Add(host_standalone_crt
      DOWNLOAD_COMMAND ""
      SOURCE_DIR standalone_crt
      CONFIGURE_COMMAND ""
      BUILD_COMMAND make
          DLPACK_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/3rdparty/dlpack/include
          TVM_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/standalone_crt/include
          CRT_CONFIG=src/runtime/crt/host/crt_config.h
          BUILD_DIR=${host_build_dir_abspath} all ${make_quiet}
     BUILD_IN_SOURCE ON
     WORKING_DIRECTORY standalone_crt
     COMMENT "Building host CRT runtime"
     BUILD_BYPRODUCTS "${crt_library_paths}"
     DEPENDS standalone_crt
     INSTALL_COMMAND ""
     )
  ExternalProject_Add_StepDependencies(host_standalone_crt build standalone_crt ${host_isolated_build_deps})

  foreach(crt_lib IN LISTS crt_libraries)
    set(cmake_crt_lib_name host_standalone_crt_${crt_lib})
    list(APPEND cmake_crt_libraries ${cmake_crt_lib_name})
    add_library(${cmake_crt_lib_name} STATIC IMPORTED GLOBAL)
    add_dependencies(${cmake_crt_lib_name} host_standalone_crt)
    set_target_properties(${cmake_crt_lib_name} PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt/lib${crt_lib}.a"
        IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt/lib${crt_lib}.a"
        PUBLIC_HEADER "${crt_headers}")
  endforeach()

  list(APPEND TVM_RUNTIME_LINKER_LIBS host_standalone_crt_utvm_rpc_server)

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
      target_include_directories(${__execname} PUBLIC ${GTEST_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/standalone_crt/include ${CMAKE_SOURCE_DIR}/src/runtime/crt/host)
      target_compile_options(${__execname} PRIVATE -pthread)
      target_link_libraries(${__execname} ${cmake_crt_libraries} ${GTEST_LIB} pthread)
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
