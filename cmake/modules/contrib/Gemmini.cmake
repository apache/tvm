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

if(USE_GEMMINI)
  message(STATUS "Add Gemmini for microTVM")

  function(microtvm_add_gemmini)
    list(
      APPEND
      GEMMINI_FILE_COPY_JOBS
      "apps/microtvm/gemmini/template_project microtvm_api_server.py -> gemmini"
      "apps/microtvm/gemmini/template_project/crt_config *.h -> gemmini/crt_config"

      # Dense example project generation
      "apps/microtvm/gemmini/template_project/src dense.c -> gemmini/src/dense_example"
      "apps/microtvm/gemmini/template_project/src Makefile -> gemmini/src/dense_example"
      "apps/microtvm/gemmini/template_project/src Makefile.in -> gemmini/src/dense_example"
      "apps/microtvm/gemmini/template_project/src Makefrag.mk -> gemmini/src/dense_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests build.sh -> gemmini/src/dense_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests configure.ac -> gemmini/src/dense_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests/include *.h -> gemmini/src/dense_example/include"
      "3rdparty/gemmini/software/gemmini-rocc-tests/rocc-software/src *.h -> gemmini/src/dense_example/rocc-software/src"

      # CONV2D example project generation
      "apps/microtvm/gemmini/template_project/src conv2d.c -> gemmini/src/conv2d_example"
      "apps/microtvm/gemmini/template_project/src Makefile -> gemmini/src/conv2d_example"
      "apps/microtvm/gemmini/template_project/src Makefile.in -> gemmini/src/conv2d_example"
      "apps/microtvm/gemmini/template_project/src Makefrag.mk -> gemmini/src/conv2d_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests build.sh -> gemmini/src/conv2d_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests configure.ac -> gemmini/src/conv2d_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests/include *.h -> gemmini/src/conv2d_example/include"
      "3rdparty/gemmini/software/gemmini-rocc-tests/rocc-software/src *.h -> gemmini/src/conv2d_example/rocc-software/src"

      # DW CONV2D example project generation
      "apps/microtvm/gemmini/template_project/src dwconv2d.c -> gemmini/src/dwconv2d_example"
      "apps/microtvm/gemmini/template_project/src Makefile -> gemmini/src/dwconv2d_example"
      "apps/microtvm/gemmini/template_project/src Makefile.in -> gemmini/src/dwconv2d_example"
      "apps/microtvm/gemmini/template_project/src Makefrag.mk -> gemmini/src/dwconv2d_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests build.sh -> gemmini/src/dwconv2d_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests configure.ac -> gemmini/src/dwconv2d_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests/include *.h -> gemmini/src/dwconv2d_example/include"
      "3rdparty/gemmini/software/gemmini-rocc-tests/rocc-software/src *.h -> gemmini/src/dwconv2d_example/rocc-software/src"

      # ADD example project generation
      "apps/microtvm/gemmini/template_project/src add.c -> gemmini/src/add_example"
      "apps/microtvm/gemmini/template_project/src Makefile -> gemmini/src/add_example"
      "apps/microtvm/gemmini/template_project/src Makefile.in -> gemmini/src/add_example"
      "apps/microtvm/gemmini/template_project/src Makefrag.mk -> gemmini/src/add_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests build.sh -> gemmini/src/add_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests configure.ac -> gemmini/src/add_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests/include *.h -> gemmini/src/add_example/include"
      "3rdparty/gemmini/software/gemmini-rocc-tests/rocc-software/src *.h -> gemmini/src/add_example/rocc-software/src"

      # Max pooling 2d example project generation
      "apps/microtvm/gemmini/template_project/src maxpool2d.c -> gemmini/src/maxpool2d_example"
      "apps/microtvm/gemmini/template_project/src Makefile -> gemmini/src/maxpool2d_example"
      "apps/microtvm/gemmini/template_project/src Makefile.in -> gemmini/src/maxpool2d_example"
      "apps/microtvm/gemmini/template_project/src Makefrag.mk -> gemmini/src/maxpool2d_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests build.sh -> gemmini/src/maxpool2d_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests configure.ac -> gemmini/src/maxpool2d_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests/include *.h -> gemmini/src/maxpool2d_example/include"
      "3rdparty/gemmini/software/gemmini-rocc-tests/rocc-software/src *.h -> gemmini/src/maxpool2d_example/rocc-software/src"

      # Mobilenet example project generation
      "apps/microtvm/gemmini/template_project/src mobilenet.c -> gemmini/src/mobilenet_example"
      "apps/microtvm/gemmini/template_project/src Makefile -> gemmini/src/mobilenet_example"
      "apps/microtvm/gemmini/template_project/src Makefile.in -> gemmini/src/mobilenet_example"
      "apps/microtvm/gemmini/template_project/src Makefrag.mk -> gemmini/src/mobilenet_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests build.sh -> gemmini/src/mobilenet_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests configure.ac -> gemmini/src/mobilenet_example"
      "3rdparty/gemmini/software/gemmini-rocc-tests/include *.h -> gemmini/src/mobilenet_example/include"
      "3rdparty/gemmini/software/gemmini-rocc-tests/rocc-software/src *.h -> gemmini/src/mobilenet_example/rocc-software/src"
    )

    foreach(job_spec IN LISTS GEMMINI_FILE_COPY_JOBS)
      string(REPLACE " " ";" job_spec "${job_spec}")
      list(LENGTH job_spec job_spec_length)
      math(EXPR job_spec_length_mod "${job_spec_length} % 3")
      if(NOT "${job_spec_length_mod}" EQUAL 1)
        message(
          FATAL_ERROR
            "Gemmini copy job spec list length is ${job_spec_length}; parsed job spec is ${job_spec}"
        )
      endif()
      math(EXPR job_spec_stop "${job_spec_length} - 3")

      list(GET job_spec 0 job_src_base)
      set(job_src_base "${CMAKE_SOURCE_DIR}/${job_src_base}")
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
              "Gemmini copy job matched 0 files: ${job_src_base}/${copy_pattern} -> ${copy_dest}"
          )
        endif()
        foreach(copy_src IN LISTS copy_files)
          get_filename_component(
            dest_path "${MICROTVM_TEMPLATE_PROJECTS}/${copy_dest}/${copy_src}"
            ABSOLUTE)
          tvm_micro_add_copy_file(gemmini_template_deps
                                  ${job_src_base}/${copy_src} ${dest_path})
        endforeach()
      endforeach()
    endforeach()

    add_custom_target(gemmini DEPENDS ${gemmini_template_deps})
  endfunction()

  microtvm_add_gemmini()

endif(USE_MICRO)
