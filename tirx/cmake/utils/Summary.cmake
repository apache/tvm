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

function(pad_string output str padchar length)
    string(LENGTH "${str}" _strlen)
    math(EXPR _strlen "${length} - ${_strlen}")

    if(_strlen GREATER 0)
        unset(_pad)
        foreach(_i RANGE 1 ${_strlen}) # inclusive
            string(APPEND _pad ${padchar})
        endforeach()
        string(APPEND str ${_pad})
    endif()

    set(${output} "${str}" PARENT_SCOPE)
endfunction()

macro(print_summary)
    message(STATUS "  ---------------- Summary ----------------")

    # Show some basic system information
    message(STATUS "  CMake version         : ${CMAKE_VERSION}")
    message(STATUS "  CMake executable      : ${CMAKE_COMMAND}")
    message(STATUS "  Generator             : ${CMAKE_GENERATOR}")
    message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler ID       : ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    message(STATUS "  CXX launcher          : ${CMAKE_CXX_COMPILER_LAUNCHER}")
    message(STATUS "  Linker flags          : ${CMAKE_SHARED_LINKER_FLAGS}")
    message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
    get_directory_property(READABLE_COMPILE_DEFS DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
    message(STATUS "  Compile definitions   : ${READABLE_COMPILE_DEFS}")

    list(SORT TVM_ALL_OPTIONS)
    message(STATUS "  Options:")

    # Compute padding necessary for options
    set(MAX_LENGTH 0)
    foreach(OPTION ${TVM_ALL_OPTIONS})
        string(LENGTH ${OPTION} OPTIONLENGTH)
        if(${OPTIONLENGTH} GREATER ${MAX_LENGTH})
            set(MAX_LENGTH ${OPTIONLENGTH})
        endif()
    endforeach()
    math(EXPR PADDING_LENGTH "${MAX_LENGTH} + 3")

    # Print each of the options (padded out so they're all aligned)
    foreach(OPTION ${TVM_ALL_OPTIONS})
        set(OPTION_VALUE "${${OPTION}}")
        pad_string(OUT "   ${OPTION}" " " ${PADDING_LENGTH})
        message(STATUS ${OUT} " : " ${OPTION_VALUE})
    endforeach()
endmacro()

function(dump_options_to_file tvm_options)
    file(REMOVE ${CMAKE_BINARY_DIR}/TVMBuildOptions.txt)
    foreach(option ${tvm_options})
        file(APPEND ${CMAKE_BINARY_DIR}/TVMBuildOptions.txt "${option} ${${option}} \n")
    endforeach()
endfunction()
