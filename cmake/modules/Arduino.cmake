# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements.  See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.  The ASF licenses this
# file to you under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

if(USE_MICRO)
  message(STATUS "Add Arduino for microTVM")

  function(microtvm_add_arduino)
    list(
      APPEND
      ARDUINO_FILE_COPY_JOBS
      "apps/microtvm/arduino/template_project microtvm_api_server.py -> arduino"
      "python/tvm/micro/project_api server.py -> arduino"
      "apps/microtvm/arduino/template_project launch_microtvm_api_server.sh -> arduino"
      "apps/microtvm/arduino/template_project boards.json -> arduino"
      "apps/microtvm/arduino/template_project/src/example_project *.c -> arduino/src/example_project"
      "apps/microtvm/arduino/template_project/src/example_project *.h -> arduino/src/example_project"
      "apps/microtvm/arduino/template_project/src/example_project *.ino -> arduino/src/example_project"
      "apps/microtvm/arduino/template_project/src/host_driven *.c -> arduino/src/host_driven"
      "apps/microtvm/arduino/template_project/src/host_driven *.ino -> arduino/src/host_driven"
      "apps/microtvm/arduino/template_project/crt_config *.h -> arduino/crt_config"
      "apps/microtvm/arduino/template_project Makefile.template -> arduino"
    )

    foreach(job_spec IN LISTS ARDUINO_FILE_COPY_JOBS)
      string(REPLACE " " ";" job_spec "${job_spec}")
      list(LENGTH job_spec job_spec_length)
      math(EXPR job_spec_length_mod "${job_spec_length} % 3")
      if(NOT "${job_spec_length_mod}" EQUAL 1)
        message(
          FATAL_ERROR
            "Arduino copy job spec list length is ${job_spec_length}; parsed job spec is ${job_spec}"
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
              "Arduino copy job matched 0 files: ${job_src_base}/${copy_pattern} -> ${copy_dest}"
          )
        endif()
        foreach(copy_src IN LISTS copy_files)
          get_filename_component(
            dest_path "${MICROTVM_TEMPLATE_PROJECTS}/${copy_dest}/${copy_src}"
            ABSOLUTE)
          tvm_micro_add_copy_file(arduino_template_deps
                                  ${job_src_base}/${copy_src} ${dest_path})
        endforeach()
      endforeach()
    endforeach()

    add_custom_target(arduino DEPENDS ${arduino_template_deps})
  endfunction()

  microtvm_add_arduino()

endif(USE_MICRO)
