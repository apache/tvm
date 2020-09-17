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

macro(__tvm_option variable description value)
  if(NOT DEFINED ${variable})
    set(${variable} ${value} CACHE STRING ${description})
  endif()
endmacro()

#######################################################
# An option that the user can select. Can accept condition to control when option is available for user.
# Usage:
#   tvm_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
macro(tvm_option variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()
  unset(__varname)
  if("${__condition}" STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if("${__value}" MATCHES ";")
      if(${__value})
        __tvm_option(${variable} "${description}" ON)
      else()
        __tvm_option(${variable} "${description}" OFF)
      endif()
    elseif(DEFINED ${__value})
      if(${__value})
        __tvm_option(${variable} "${description}" ON)
      else()
        __tvm_option(${variable} "${description}" OFF)
      endif()
    else()
      __tvm_option(${variable} "${description}" "${__value}")
    endif()
  else()
    unset(${variable} CACHE)
  endif()
endmacro()

function(assign_source_group group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${group}\\${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)

cmake_policy(SET CMP0057 NEW) # Needed for IN_LIST used in conditional
function(DECOMPOSE_ARG SOURCE_VALUE)
  set(options)
  set(oneValueArgs ENABLE_VALUE COMMAND_VALUE)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  # First check if SOURCE_VALUE is executable
  execute_process(COMMAND ${SOURCE_VALUE}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE dummy
    OUTPUT_QUIET ERROR_QUIET)
  string(TOUPPER ${SOURCE_VALUE} UPPER_CASE_SOURCE_VALUE)
  set(FALSE_VALUES OFF 0 NO FALSE N IGNORE NOTFOUND)
  if(${SOURCE_VALUE})
    # If not executable, is it true
    set(${ARG_ENABLE_VALUE} ON PARENT_SCOPE)
    unset(${ARG_COMMAND_VALUE} PARENT_SCOPE)
  elseif(${UPPER_CASE_SOURCE_VALUE} IN_LIST FALSE_VALUES)
    set(${ARG_ENABLE_VALUE} OFF PARENT_SCOPE)
    unset(${ARG_COMMAND_VALUE} PARENT_SCOPE)
  elseif(${exit_code} MATCHES [0-9]+)
    set(${ARG_ENABLE_VALUE} ON PARENT_SCOPE)
    set(${ARG_COMMAND_VALUE} ${SOURCE_VALUE} PARENT_SCOPE)
  else()
    message(FATAL_ERROR "argument for cmake flag neither a command nor a boolean")
  endif()
endfunction(DECOMPOSE_ARG)
