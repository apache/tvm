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
  include(cmake/utils/CRTConfig.cmake)

  message(STATUS "Build with Micro support")
  tvm_file_glob(GLOB RUNTIME_MICRO_SRCS src/runtime/micro/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_MICRO_SRCS})

  function(microtvm_add_platform_project_api platform)
    if("${platform}" STREQUAL "zephyr")
      list(
        APPEND
        PLATFORM_FILE_COPY_JOBS
        "apps/microtvm/zephyr/template_project microtvm_api_server.py -> zephyr"
        "python/tvm/micro/project_api server.py -> zephyr"
        "apps/microtvm/zephyr/template_project launch_microtvm_api_server.sh -> zephyr"
        "apps/microtvm/zephyr/template_project boards.json -> zephyr"
        "apps/microtvm/zephyr/template_project CMakeLists.txt.template -> zephyr"
        "apps/microtvm/zephyr/template_project/src/aot_standalone_demo *.c -> zephyr/src/aot_standalone_demo"
        "apps/microtvm/zephyr/template_project/src/host_driven *.c -> zephyr/src/host_driven"
        "apps/microtvm/zephyr/template_project/src/host_driven *.h -> zephyr/src/host_driven"
        "apps/microtvm/zephyr/template_project/src/mlperftiny *.cc -> zephyr/src/mlperftiny"
        "3rdparty/mlperftiny/api * -> zephyr/src/mlperftiny/api"
        "apps/microtvm/zephyr/template_project/fvp-hack * -> zephyr/fvp-hack"
        "apps/microtvm/zephyr/template_project/qemu-hack * -> zephyr/qemu-hack"
        "apps/microtvm/zephyr/template_project/app-overlay * -> zephyr/app-overlay"
      )
    elseif("${platform}" STREQUAL "arduino")
      list(
        APPEND
        PLATFORM_FILE_COPY_JOBS
        "apps/microtvm/arduino/template_project microtvm_api_server.py -> arduino"
        "python/tvm/micro/project_api server.py -> arduino"
        "apps/microtvm/arduino/template_project launch_microtvm_api_server.sh -> arduino"
        "apps/microtvm/arduino/template_project boards.json -> arduino"
        "apps/microtvm/arduino/template_project/src/example_project *.c -> arduino/src/example_project"
        "apps/microtvm/arduino/template_project/src/example_project *.h -> arduino/src/example_project"
        "apps/microtvm/arduino/template_project/src/example_project *.ino -> arduino/src/example_project"
        "apps/microtvm/arduino/template_project/src/host_driven *.c -> arduino/src/host_driven"
        "apps/microtvm/arduino/template_project/src/host_driven *.ino -> arduino/src/host_driven"
        "apps/microtvm/arduino/template_project Makefile.template -> arduino"
      )
    elseif("${platform}" STREQUAL "crt")
      list(
        APPEND
        PLATFORM_FILE_COPY_JOBS
        "src/runtime/crt/host microtvm_api_server.py -> crt"
        "src/runtime/crt/host CMakeLists.txt.template -> crt"
        "src/runtime/crt/host **.cc -> crt/src"
      )
    else()
      message(FATAL_ERROR "${platform} not supported.")
    endif()

    foreach(job_spec IN LISTS PLATFORM_FILE_COPY_JOBS)
      string(REPLACE " " ";" job_spec "${job_spec}")
      list(LENGTH job_spec job_spec_length)
      math(EXPR job_spec_length_mod "${job_spec_length} % 3")
      if(NOT "${job_spec_length_mod}" EQUAL 1)
        message(
          FATAL_ERROR
            "${platform} copy job spec list length is ${job_spec_length}; parsed job spec is ${job_spec}"
        )
      endif()
      math(EXPR job_spec_stop "${job_spec_length} - 3")

      list(GET job_spec 0 job_src_base)
      set(job_src_base "${CMAKE_CURRENT_SOURCE_DIR}/${job_src_base}")
      foreach(copy_pattern_index RANGE 1 "${job_spec_stop}" 3)
        list(GET job_spec ${copy_pattern_index} copy_pattern)
        math(EXPR copy_dest_index "${copy_pattern_index} + 2")
        list(GET job_spec ${copy_dest_index} copy_dest)

        file(
          GLOB_RECURSE copy_files
          RELATIVE "${job_src_base}"
          "${job_src_base}/${copy_pattern}")
        list(LENGTH copy_files copy_files_length)
        if("${copy_files_length}" EQUAL 0)
          message(
            FATAL_ERROR
              "${platform} copy job matched 0 files: ${job_src_base}/${copy_pattern} -> ${copy_dest}"
          )
        endif()
        foreach(copy_src IN LISTS copy_files)
          get_filename_component(
            dest_path "${MICROTVM_TEMPLATE_PROJECTS}/${copy_dest}/${copy_src}"
            ABSOLUTE)
          tvm_micro_add_copy_file(platform_template_deps
                                  ${job_src_base}/${copy_src} ${dest_path})
        endforeach()
      endforeach()
    endforeach()

    add_custom_target(${platform} DEPENDS ${platform_template_deps})
  endfunction()

  set(PLATFORMS crt;zephyr;arduino)
  foreach(platform IN LISTS PLATFORMS)
    message(STATUS "Add ${platform} template project.")
    microtvm_add_platform_project_api(${platform})
    generate_crt_config(${platform} "${CMAKE_CURRENT_BINARY_DIR}/microtvm_template_projects/${platform}/crt_config/crt_config.h")
  endforeach()

  # Add template files for Model Library Format
  generate_crt_config("crt" "${MICROTVM_TEMPLATE_PROJECTS}/crt/templates/crt_config.h.template")
  configure_file("src/runtime/crt/platform-template.c" "${MICROTVM_TEMPLATE_PROJECTS}/crt/templates/platform.c.template" COPYONLY)
endif(USE_MICRO)
