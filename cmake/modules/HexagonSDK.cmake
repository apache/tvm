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

macro(set_parent _var)
  set(${_var} ${ARGN} PARENT_SCOPE)
endmacro()

# Check if the path in _path exists. If true, set _output_variable
# to the absolute path, otherwise set it to the value of _path with
# the suffix "-NOTFOUND" appended to it.
function(_check_path_exists _path _output_variable)
  file(TO_NATIVE_PATH "/" _native_root_dir)
  file(RELATIVE_PATH _absolute_path "${_native_root_dir}" "${_path}")
  # RELATIVE_PATH will strip the root, so add it back.
  set(_absolute_path "${_native_root_dir}${_absolute_path}")
  if(EXISTS "${_absolute_path}")
    set_parent(${_output_variable} "${_absolute_path}")
  else()
    set_parent(${_output_variable} "${_path}-NOTFOUND")
  endif()
endfunction()

# Check if all paths in the _paths list exist. If so set _output_variable
# to the list of the absolute paths from the input list, otherwise set
# _output_variable to the first path that was not found, and append the
# "-NOTFOUND" suffix to it.
function(_check_all_paths_exist _paths _output_variable)
  foreach(_path IN LISTS _paths)
    _check_path_exists("${_path}" _out_path)
    if(_out_path)
      list(APPEND _out_paths "${_out_path}")
    else()
      set_parent(${_output_variable} "${_path}-NOTFOUND")
      return()
    endif()
  endforeach()
  set_parent(${_output_variable} ${_out_paths})
endfunction()

function(_get_linux_version _output_vendor _output_release)
  execute_process(
    COMMAND lsb_release "-is"
    OUTPUT_VARIABLE _vendor
  )
  if(_vendor)
    string(STRIP "${_vendor}" _vendor)
    set_parent(${_output_vendor} ${_vendor})
  else()
    set_parent(${_output_vendor} "NOTFOUND")
  endif()
  execute_process(
    COMMAND lsb_release "-rs"
    OUTPUT_VARIABLE _release
  )
  if(_release)
    string(STRIP "${_release}" _release)
    set_parent(${_output_release} ${_release})
  else()
    set_parent(${_output_release} "NOTFOUND")
  endif()
endfunction()

function(_get_ubuntu_version _output_version)
  _get_linux_version(_vendor _release)
  if(_vendor STREQUAL "Ubuntu")
    string(REGEX MATCH "[0-9]+" _release_major "${_release}")
    if(_release_major)
      set_parent(${_output_version} "${_vendor}${_release_major}")
      return()
    endif()
  endif()
  set_parent(${_output_version} "NOTFOUND")
endfunction()

function(_get_hexagon_sdk_property_impl
         _hexagon_sdk_root _hexagon_arch _property _output_variable)
  # Properties
  #   VERSION
  #   SDK_INCLUDE
  #   QURT_INCLUDE
  #   QURT_LIB
  #   RPCMEM_ROOT
  #   DSPRPC_LIB
  #   QAIC_EXE

  if(${ARGC} LESS "4")
    message(FATAL_ERROR
      "Invalid number of arguments to get_hexagon_sdk_property"
    )
  endif()

  # Set the Hexagon arch directory component.
  set(_hexarch_dir_v65 "computev65")
  set(_hexarch_dir_v66 "computev66")
  set(_hexarch_dir_v68 "computev68")
  set(_hexarch_dir_v69 "computev69")
  set(_hexarch_dir_v73 "computev73")
  set(_hexarch_dir_v75 "computev75")
  set(_hexarch_dir_str "_hexarch_dir_${_hexagon_arch}")
  set(_hexarch_dir "${${_hexarch_dir_str}}")

  if(NOT _hexarch_dir)
    message(SEND_ERROR "Please set Hexagon architecture to one of v65, v66, v68, v69, v73, v75")
  endif()

  if(_property STREQUAL "VERSION")
    _check_path_exists("${_hexagon_sdk_root}/incs/version.h" _version_header)
    if(_version_header)
      execute_process(
        COMMAND grep "#define[ \t]*VERSION_STRING" "${_version_header}"
        OUTPUT_VARIABLE _version_define
      )
      string(
        REGEX REPLACE ".*VERSION_STRING.* ([0-9\\.]+) .*" "\\1"
        _version_string "${_version_define}"
      )
      set_parent(${_output_variable} ${_version_string})
    else()
      set_parent(${_output_variable} "${_property}-NOTFOUND")
    endif()

  elseif(_property STREQUAL "QAIC_EXE")
    set(_override $ENV{QAIC_PATH_OVERRIDE})
    if(_override)
      _check_path_exists("${_override}" _qaic_path)
    else()
      _get_ubuntu_version(_uversion)
      _check_path_exists(
        "${_hexagon_sdk_root}/ipc/fastrpc/qaic/${_uversion}/qaic"
        _qaic_path
      )
    endif()
    if(NOT _qaic_path)
      message(
        WARNING
        "The qaic executable cannot be found in '${_qaic_path}'. You can set "
        "the environment variable QAIC_PATH_OVERRIDE to override the automatic "
        "search."
      )
    endif()
    set_parent(${_output_variable} "${_qaic_path}")

  else()
    # The rest of the checks returns path(s), which shares some common code.
    if(_property STREQUAL "SDK_INCLUDE")
      set(_dirs "${_hexagon_sdk_root}/incs" "${_hexagon_sdk_root}/incs/stddef")
    elseif(_property STREQUAL "QURT_INCLUDE")
      # Set the Hexagon arch directory for runtime linker.
      set(_rtld_dir "hexagon_toolv84_${_hexagon_arch}")
      if(_hexagon_arch STREQUAL "v75")
        set(_rtld_dir "hexagon_toolv87_v75") # Use hexagon_toolv87_v75 for v75
      endif()
      if(_hexagon_arch STREQUAL "v69")
        set(_rtld_dir "hexagon_toolv84_v68") # Use hexagon_toolv84_v68 for v69
      endif()
      set(_dirs
        "${_hexagon_sdk_root}/rtos/qurt/${_hexarch_dir}/include/posix"
        "${_hexagon_sdk_root}/rtos/qurt/${_hexarch_dir}/include/qurt"
        "${_hexagon_sdk_root}/ipc/fastrpc/rtld/ship/${_rtld_dir}"
      )
      _check_path_exists("${_hexagon_sdk_root}/ipc/fastrpc/rtld/ship/inc" _sdk_dlfcn)
      if(_sdk_dlfcn)
        list(APPEND _dirs "${_hexagon_sdk_root}/ipc/fastrpc/rtld/ship/inc")
      endif()
    elseif(_property STREQUAL "QURT_LIB")
      set(_dirs "${_hexagon_sdk_root}/rtos/qurt/${_hexarch_dir}/lib/pic")
    elseif(_property STREQUAL "RPCMEM_ROOT")
      set(_dirs "${_hexagon_sdk_root}/ipc/fastrpc/rpcmem")
    elseif(_property STREQUAL "DSPRPC_LIB")
      set(_dirs "${_hexagon_sdk_root}/ipc/fastrpc/remote/ship/android_aarch64")
    else()
      message(SEND_ERROR "Unknown SDK property ${_property}")
    endif()

    _check_all_paths_exist("${_dirs}" _dirs_exist)
    set_parent(${_output_variable} "${_dirs_exist}")
  endif()
endfunction()

function(get_hexagon_sdk_property _hexagon_sdk_root _hexagon_arch)
  math(EXPR _pnum ${ARGC}-3)      # _pnum = number of extra arguments minus 1
  foreach(_p RANGE 0 ${_pnum} 2)  # Range includes the upper bound.
    list(GET ARGN 0 _property)
    list(GET ARGN 1 _outvar)
    _get_hexagon_sdk_property_impl(${_hexagon_sdk_root} ${_hexagon_arch}
      ${_property} _out
    )
    set_parent(${_outvar} ${_out})
    list(REMOVE_AT ARGN 0 1)
  endforeach()
endfunction()
