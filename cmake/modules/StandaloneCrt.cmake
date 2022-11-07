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


if(USE_MICRO)

if(MSVC)

  # When building for Windows, use standard CMake for compatibility with
  # Visual Studio build tools and not require Make to be on the system.

  set(CRT_CONFIG, "src/runtime/micro/crt_config.h")

  add_library(host_standalone_crt
              STATIC
              3rdparty/libcrc/src/crcccitt.c
              src/runtime/crt/microtvm_rpc_common/frame_buffer.cc
              src/runtime/crt/microtvm_rpc_common/framing.cc
              src/runtime/crt/microtvm_rpc_common/session.cc
              src/runtime/crt/microtvm_rpc_common/write_stream.cc)

  target_include_directories(host_standalone_crt
                             PRIVATE
                             3rdparty/libcrc/include
                             src/runtime/micro)

else()

  message(STATUS "Build standalone CRT for microTVM")

  function(tvm_crt_define_targets)
    # Build an isolated build directory, separate from the TVM tree.
    list(APPEND CRT_FILE_COPY_JOBS
         "3rdparty/libcrc/include *.h -> include"
         "3rdparty/libcrc/src crcccitt.c -> src/runtime/crt/microtvm_rpc_common"
         "3rdparty/libcrc/tab gentab_ccitt.inc -> src/runtime/crt/tab"
         "3rdparty/dlpack/include *.h -> include"
         "3rdparty/dmlc-core/include *.h -> include"
         "include/tvm/runtime c_*_api.h -> include/tvm/runtime"
         "include/tvm/runtime metadata_types.h -> include/tvm/runtime"
         "include/tvm/runtime/crt *.h -> include/tvm/runtime/crt"
         "src/runtime/crt Makefile -> ."
         "src/runtime/crt/include *.h -> include"
         "src/runtime/crt/aot_executor *.c -> src/runtime/crt/aot_executor"
         "src/runtime/crt/aot_executor_module *.c -> src/runtime/crt/aot_executor_module"
         "src/runtime/crt/common *.c -> src/runtime/crt/common"
         "src/runtime/crt/graph_executor *.c -> src/runtime/crt/graph_executor"
         "src/runtime/crt/graph_executor_module *.c -> src/runtime/crt/graph_executor_module"
         "src/runtime/crt/host *.cc -> template/host"
         "src/runtime/crt/host *.py -> template/host"
         "src/runtime/crt/host Makefile -> template/host"
         "src/runtime/crt/memory *.c -> src/runtime/crt/memory"
         "src/runtime/crt/microtvm_rpc_common *.cc -> src/runtime/crt/microtvm_rpc_common"
         "src/runtime/crt/microtvm_rpc_server *.cc -> src/runtime/crt/microtvm_rpc_server"
         "src/runtime/minrpc *.h -> src/runtime/minrpc"
         "src/support generic_arena.h -> src/support"
         "src/support ssize.h -> src/support"
         "src/runtime/crt crt_config-template.h -> template"
         )

    set(standalone_crt_base "${CMAKE_CURRENT_BINARY_DIR}/standalone_crt")

    foreach(job_spec IN LISTS CRT_FILE_COPY_JOBS)
      string(REPLACE " " ";" job_spec "${job_spec}")
      list(LENGTH job_spec job_spec_length)
      math(EXPR job_spec_length_mod "${job_spec_length} % 3")
      if(NOT "${job_spec_length_mod}" EQUAL 1)
        message(FATAL_ERROR "CRT copy job spec list length is ${job_spec_length}; parsed job spec is ${job_spec}")
      endif()
      math(EXPR job_spec_stop "${job_spec_length} - 3")

      list(GET job_spec 0 job_src_base)
      set(job_src_base "${CMAKE_CURRENT_SOURCE_DIR}/${job_src_base}")
      foreach(copy_pattern_index RANGE 1 "${job_spec_stop}" 3)
        list(GET job_spec ${copy_pattern_index} copy_pattern)
        math(EXPR copy_dest_index "${copy_pattern_index} + 2")
        list(GET job_spec ${copy_dest_index} copy_dest)

        tvm_file_glob(GLOB_RECURSE copy_files
             RELATIVE "${job_src_base}"
             "${job_src_base}/${copy_pattern}")
        list(LENGTH copy_files copy_files_length)
        if("${copy_files_length}" EQUAL 0)
          message(FATAL_ERROR "CRT copy job matched 0 files: ${job_src_base}/${copy_pattern} -> ${copy_dest}")
        endif()
        foreach(copy_src IN LISTS copy_files)
          get_filename_component(dest_path "${standalone_crt_base}/${copy_dest}/${copy_src}" ABSOLUTE)
          tvm_micro_add_copy_file(host_isolated_build_deps ${job_src_base}/${copy_src} ${dest_path})
        endforeach()
      endforeach()
    endforeach()

    add_custom_target(standalone_crt DEPENDS ${host_isolated_build_deps})

    get_filename_component(host_build_dir_abspath "${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt" ABSOLUTE)

    if(${VERBOSE})
    set(make_quiet QUIET=)
    else(${VERBOSE})
    set(make_quiet )
    endif(${VERBOSE})

    list(APPEND crt_libraries memory graph_executor microtvm_rpc_server microtvm_rpc_common common)  # NOTE: listed in link order.
    foreach(crt_lib_name IN LISTS crt_libraries)
      list(APPEND crt_library_paths "host_standalone_crt/lib${crt_lib_name}.a")
    endforeach()

    set(make_common_args
        "CRT_CONFIG=${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/micro/crt_config.h"
        "BUILD_DIR=${host_build_dir_abspath}"
        "EXTRA_CFLAGS=-fPIC"
        "EXTRA_CXXFLAGS=-fPIC"
        "EXTRA_LDFLAGS=-fPIC"
        "${make_quiet}")

    add_custom_command(
          OUTPUT ${crt_library_paths}
          COMMAND make ARGS ${make_common_args} clean
          COMMAND make ARGS ${make_common_args} all
          WORKING_DIRECTORY "${standalone_crt_base}"
          DEPENDS standalone_crt ${host_isolated_build_deps})

    add_custom_target(host_standalone_crt DEPENDS ${crt_library_paths})

    foreach(crt_lib IN LISTS crt_libraries)
      set(cmake_crt_lib_name host_standalone_crt_${crt_lib})
      list(APPEND cmake_crt_libraries ${cmake_crt_lib_name})
      add_library(${cmake_crt_lib_name} STATIC IMPORTED GLOBAL)
      set(cmake_crt_lib_path "${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt/lib${crt_lib}.a")
      add_dependencies(${cmake_crt_lib_name} host_standalone_crt "${cmake_crt_lib_path}")
      set_target_properties(${cmake_crt_lib_name} PROPERTIES
          IMPORTED_LOCATION "${cmake_crt_lib_path}"
          IMPORTED_OBJECTS "${cmake_crt_lib_path}"
          PUBLIC_HEADER "${crt_headers}")
    endforeach()

    # Create the `crttest` target if we can find GTest.  If not, we create dummy
    # targets that give the user an informative error message.
    if(GTEST_FOUND)
      tvm_file_glob(GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/tests/crt/*.cc)
      add_executable(crttest ${TEST_SRCS})
      target_include_directories(crttest SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/standalone_crt/include ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/micro)
      target_link_libraries(crttest PRIVATE ${cmake_crt_libraries} GTest::GTest GTest::Main pthread dl)
      set_target_properties(crttest PROPERTIES EXCLUDE_FROM_ALL 1)
      set_target_properties(crttest PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
      gtest_discover_tests(crttest)
    endif()

  endfunction()

  tvm_crt_define_targets()

  set(TVM_CRT_LINKER_LIB host_standalone_crt_microtvm_rpc_common)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(APPEND TVM_RUNTIME_LINKER_LIBS -Wl,--whole-archive ${TVM_CRT_LINKER_LIB} -Wl,--no-whole-archive)
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang")
  list(APPEND TVM_RUNTIME_LINKER_LIBS -Wl,-force_load $<TARGET_PROPERTY:${TVM_CRT_LINKER_LIB},IMPORTED_LOCATION>)
  else()
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${TVM_CRT_LINKER_LIB})
  endif()

endif()

endif()
