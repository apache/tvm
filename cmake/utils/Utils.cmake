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

set(TVM_ALL_OPTIONS)

#######################################################
# An option that the user can select. Can accept condition to control when option is available for user.
# Usage:
#   tvm_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
macro(tvm_option variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  list(APPEND TVM_ALL_OPTIONS ${variable})
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
      # list values directly pass through
      __tvm_option(${variable} "${description}" "${__value}")
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

# From cmake documentation:
# True if the constant is 1, ON, YES, TRUE, Y, or a non-zero number.
# False if the constant is 0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, the empty string, or ends in the suffix -NOTFOUND.
# Named boolean constants are case-insensitive.
#
# While this regex does contain a check for an empty string that check does not work
# cmake's regex is weak
set(IS_FALSE_PATTERN "^[Oo][Ff][Ff]$|^0$|^[Ff][Aa][Ll][Ss][Ee]$|^[Nn][Oo]$|^[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|.*-[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|^$")
set(IS_TRUE_PATTERN "^[Oo][Nn]$|^[1-9][0-9]*$|^[Tt][Rr][Uu][Ee]$|^[Yy][Ee][Ss]$|^[Yy]$")

# Custom file() macro that automatically uses CONFIGURE_DEPENDS if cmake
# supports it. CONFIGURE_DEPENDS scans the globbed directories on each build to
# check if any files have been added/removed. This has a small build overhead,
# but ensures that you don't manually have to rerun cmake if files were added.
# The macro should be used like so:
# tvm_file_glob(GLOB VARIABLE_NAME dir/*.cc dir/*c)
# or
# tvm_file_glob(GLOB_RECURSE VARIABLE_NAME dir/*/*.cc)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  macro(tvm_file_glob glob variable)
    file(${glob} ${variable} CONFIGURE_DEPENDS ${ARGN})
  endmacro()
else()
  macro(tvm_file_glob)
    file(${glob} ${variable} ${ARGN})
  endmacro()
endif()
