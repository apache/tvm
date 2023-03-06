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

  message(STATUS "Build microTVM RPC common")

  include(cmake/utils/CRTConfig.cmake)
  set(CRT_CONFIG_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR}/crt_config)
  generate_crt_config("crt" "${CRT_CONFIG_INCLUDE_PATH}/crt_config.h")

  # add microTVM RPC common files to TVM runtime build
  list(APPEND TVM_CRT_SOURCES
      3rdparty/libcrc/src/crcccitt.c
      src/runtime/crt/microtvm_rpc_common/frame_buffer.cc
      src/runtime/crt/microtvm_rpc_common/framing.cc
      src/runtime/crt/microtvm_rpc_common/session.cc
      src/runtime/crt/microtvm_rpc_common/write_stream.cc)

  list(APPEND RUNTIME_SRCS ${TVM_CRT_SOURCES})
  include_directories(SYSTEM ${CRT_CONFIG_INCLUDE_PATH})


  function(create_crt_library CRT_LIBRARY)

    set(CRT_LIBRARY_NAME host_standalone_crt_${CRT_LIBRARY})
    set(CRT_LIBRARY_SOURCES "")

    foreach(FILE_NAME IN LISTS ARGN)
      list(APPEND CRT_LIBRARY_SOURCES ${FILE_NAME})
    endforeach()

    add_library(${CRT_LIBRARY_NAME}
                STATIC
                ${CRT_LIBRARY_SOURCES})

    # add this library to the list of CRT libraries
    set(CRT_LIBRARIES ${CRT_LIBRARIES} ${CRT_LIBRARY_NAME} PARENT_SCOPE)

    target_include_directories(${CRT_LIBRARY_NAME}
                              PUBLIC
                              ${CRT_CONFIG_INCLUDE_PATH}
                              ${STANDALONE_CRT_BASE}/include)

    set_target_properties(${CRT_LIBRARY_NAME}
                          PROPERTIES
                          ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/host_standalone_crt
                          POSITION_INDEPENDENT_CODE ON)

    # make these libraries dependent on standalone_crt which depends on host_isolated_build_deps to avoid
    # race with the file copy jobs
    add_dependencies(${CRT_LIBRARY_NAME} standalone_crt)

  endfunction()

  message(STATUS "Build microTVM standalone CRT")

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
        "src/runtime/crt CMakeLists.txt -> ."
        "src/runtime/crt/include *.h -> include"
        "src/runtime/crt/aot_executor *.c -> src/runtime/crt/aot_executor"
        "src/runtime/crt/aot_executor_module *.c -> src/runtime/crt/aot_executor_module"
        "src/runtime/crt/common *.c -> src/runtime/crt/common"
        "src/runtime/crt/graph_executor *.c -> src/runtime/crt/graph_executor"
        "src/runtime/crt/graph_executor_module *.c -> src/runtime/crt/graph_executor_module"
        "src/runtime/crt/memory *.c -> src/runtime/crt/memory"
        "src/runtime/crt/microtvm_rpc_common *.cc -> src/runtime/crt/microtvm_rpc_common"
        "src/runtime/crt/microtvm_rpc_server *.cc -> src/runtime/crt/microtvm_rpc_server"
        "src/runtime/minrpc *.h -> src/runtime/minrpc"
        "src/support generic_arena.h -> src/support"
        "src/support ssize.h -> src/support"
        )

  set(STANDALONE_CRT_BASE ${CMAKE_CURRENT_BINARY_DIR}/standalone_crt)

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
        get_filename_component(dest_path "${STANDALONE_CRT_BASE}/${copy_dest}/${copy_src}" ABSOLUTE)
        tvm_micro_add_copy_file(host_isolated_build_deps ${job_src_base}/${copy_src} ${dest_path})
      endforeach()
    endforeach()
  endforeach()

  add_custom_target(standalone_crt DEPENDS ${host_isolated_build_deps})

  set(CRT_LIBRARIES "")
  set(RUNTIME_CRT_SOURCE_DIR ${STANDALONE_CRT_BASE}/src/runtime/crt)

  # these create_crt_library() targets are in link order and the common library needs to be last
  create_crt_library(aot_executor
                    ${RUNTIME_CRT_SOURCE_DIR}/aot_executor/aot_executor.c)

  create_crt_library(aot_executor_module
                    ${RUNTIME_CRT_SOURCE_DIR}/aot_executor_module/aot_executor_module.c)

  create_crt_library(graph_executor
                    ${RUNTIME_CRT_SOURCE_DIR}/graph_executor/graph_executor.c
                    ${RUNTIME_CRT_SOURCE_DIR}/graph_executor/load_json.c)

  create_crt_library(graph_executor_module
                    ${RUNTIME_CRT_SOURCE_DIR}/graph_executor_module/graph_executor_module.c)

  create_crt_library(memory
                    ${RUNTIME_CRT_SOURCE_DIR}/memory/page_allocator.c
                    ${RUNTIME_CRT_SOURCE_DIR}/memory/stack_allocator.c)

  create_crt_library(microtvm_rpc_common
                    ${RUNTIME_CRT_SOURCE_DIR}/microtvm_rpc_common/crcccitt.c
                    ${RUNTIME_CRT_SOURCE_DIR}/microtvm_rpc_common/frame_buffer.cc
                    ${RUNTIME_CRT_SOURCE_DIR}/microtvm_rpc_common/framing.cc
                    ${RUNTIME_CRT_SOURCE_DIR}/microtvm_rpc_common/session.cc
                    ${RUNTIME_CRT_SOURCE_DIR}/microtvm_rpc_common/write_stream.cc)

  create_crt_library(microtvm_rpc_server
                    ${RUNTIME_CRT_SOURCE_DIR}/microtvm_rpc_server/rpc_server.cc)

  if(NOT MSVC)
    # TODO: if we want to eventually build standalone_crt for windows
    # these files would be needed, but for now don't build them
    create_crt_library(common
                      ${RUNTIME_CRT_SOURCE_DIR}/common/crt_backend_api.c
                      ${RUNTIME_CRT_SOURCE_DIR}/common/crt_runtime_api.c
                      ${RUNTIME_CRT_SOURCE_DIR}/common/func_registry.c
                      ${RUNTIME_CRT_SOURCE_DIR}/common/ndarray.c
                      ${RUNTIME_CRT_SOURCE_DIR}/common/packed_func.c)
  endif()

  add_custom_target(host_standalone_crt DEPENDS ${CRT_LIBRARIES} standalone_crt)

  # Create the `crttest` target if we can find GTest.  If not, we create dummy
  # targets that give the user an informative error message.
  if(GTEST_FOUND)
    tvm_file_glob(GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/tests/crt/*.cc)
    add_executable(crttest ${TEST_SRCS})
    target_include_directories(crttest SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/standalone_crt/include ${CMAKE_CURRENT_BINARY_DIR}/crt_config)
    target_link_libraries(crttest PRIVATE ${CRT_LIBRARIES} GTest::GTest GTest::Main pthread dl)
    set_target_properties(crttest PROPERTIES EXCLUDE_FROM_ALL 1)
    set_target_properties(crttest PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
    gtest_discover_tests(crttest)
  endif()

endif()
