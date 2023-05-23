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

if(USE_CCACHE) # True for AUTO, ON, /path/to/ccache
  if(DEFINED CMAKE_CXX_COMPILER_LAUNCHER OR DEFINED CMAKE_C_COMPILER_LAUNCHER)
    if("${USE_CCACHE}" STREQUAL "AUTO")
      message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER or CMAKE_C_COMPILER_LAUNCHER already defined, not using ccache")
    elseif("${USE_CCACHE}" MATCHES ${IS_TRUE_PATTERN})
      message(FATAL_ERROR "CMAKE_CXX_COMPILER_LAUNCHER or CMAKE_C_COMPILER_LAUNCHER is already defined, refusing to override with ccache. Either unset or disable ccache.")
    endif()
  else()
    if("${USE_CCACHE}" STREQUAL "AUTO") # Auto mode
      find_program(CCACHE_FOUND "ccache")
      if(CCACHE_FOUND)
        message(STATUS "Found the path to ccache, enabling ccache")
        set(PATH_TO_CCACHE "ccache")
      else()
        message(STATUS "Didn't find the path to CCACHE, disabling ccache")
      endif(CCACHE_FOUND)
    elseif("${USE_CCACHE}" MATCHES ${IS_TRUE_PATTERN})
      find_program(CCACHE_FOUND "ccache")
      if(CCACHE_FOUND)
        message(STATUS "Found the path to ccache, enabling ccache")
        set(PATH_TO_CCACHE "ccache")
      else()
        message(FATAL_ERROR "Cannot find ccache. Set USE_CCACHE mode to AUTO or OFF to build without ccache. USE_CCACHE=" "${USE_CCACHE}")
      endif(CCACHE_FOUND)
    else() # /path/to/ccache
      set(PATH_TO_CCACHE "${USE_CCACHE}")
      message(STATUS "Setting ccache path to " "${PATH_TO_CCACHE}")
    endif()
    # Set the flag for ccache
    if(DEFINED PATH_TO_CCACHE)
      set(CMAKE_CXX_COMPILER_LAUNCHER "${PATH_TO_CCACHE}")
      set(CMAKE_C_COMPILER_LAUNCHER "${PATH_TO_CCACHE}")
    endif()
  endif()
endif(USE_CCACHE)
