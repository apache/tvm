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

#######################################################
# Enhanced version of find OpenCL.
#
# Usage:
#   find_opencl(${USE_OPENCL})
#
# - When USE_OPENCL=ON, use OpenCL wrapper for dynamic linking
# - When USE_OPENCL=/path/to/opencl-sdk-path, use the sdk.
#   Can be useful when cross compiling and cannot rely on
#   CMake to provide the correct library as part of the
#   requested toolchain.
#
# Provide variables:
#
# - OpenCL_FOUND
# - OpenCL_INCLUDE_DIRS
# - OpenCL_LIBRARIES
#

macro(find_opencl use_opencl)
  set(__use_opencl ${use_opencl})
  if(IS_DIRECTORY ${__use_opencl})
    set(__opencl_sdk ${__use_opencl})
    message(STATUS "Custom OpenCL SDK PATH=" ${__use_opencl})
   elseif(IS_DIRECTORY $ENV{OPENCL_SDK})
     set(__opencl_sdk $ENV{OPENCL_SDK})
   else()
     set(__opencl_sdk "")
   endif()

   if(__opencl_sdk)
     set(OpenCL_INCLUDE_DIRS ${__opencl_sdk}/include ${__opencl_sdk})
     if (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY STREQUAL "ONLY")
       set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
     endif()
     # we are in the section dedicated to the explicit pointing of OpenCL SDK path, we must not
     # look for the OpenCL library by default path, but should be limited by provided SDK
     find_library(OpenCL_LIBRARIES NAMES OpenCL NO_DEFAULT_PATH PATHS ${__opencl_sdk}/lib ${__opencl_sdk}/lib64 ${__opencl_sdk}/lib/x64/)
     if(OpenCL_LIBRARIES)
       set(OpenCL_FOUND TRUE)
     endif()
   endif(__opencl_sdk)

   # No user provided OpenCL include/libs found
   if(NOT OpenCL_FOUND)
     if(${__use_opencl} MATCHES ${IS_TRUE_PATTERN})
       find_package(OpenCL QUIET)
     endif()
   endif()

  if(OpenCL_FOUND)
    message(STATUS "OpenCL_INCLUDE_DIRS=" ${OpenCL_INCLUDE_DIRS})
    message(STATUS "OpenCL_LIBRARIES=" ${OpenCL_LIBRARIES})
  endif(OpenCL_FOUND)
endmacro(find_opencl)
