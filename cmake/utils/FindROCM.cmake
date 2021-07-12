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
# Enhanced version of find rocm.
#
# Usage:
#   find_rocm(${USE_ROCM})
#
# - When USE_ROCM=ON, use auto search
# - When USE_ROCM=/path/to/rocm-sdk-path, use the sdk
#
# Provide variables:
#
# - ROCM_FOUND
# - ROCM_INCLUDE_DIRS
# - ROCM_HIPHCC_LIBRARY
# - ROCM_MIOPEN_LIBRARY
# - ROCM_ROCBLAS_LIBRARY
#

macro(find_rocm use_rocm)
  set(__use_rocm ${use_rocm})
  if(IS_DIRECTORY ${__use_rocm})
    set(__rocm_sdk ${__use_rocm})
    message(STATUS "Custom ROCM SDK PATH=" ${__use_rocm})
  elseif(IS_DIRECTORY $ENV{ROCM_PATH})
    set(__rocm_sdk $ENV{ROCM_PATH})
  elseif(IS_DIRECTORY /opt/rocm)
    set(__rocm_sdk /opt/rocm)
  else()
    set(__rocm_sdk "")
  endif()

  if(__rocm_sdk)
    set(ROCM_INCLUDE_DIRS ${__rocm_sdk}/include)
    find_library(ROCM_HIPHCC_LIBRARY amdhip64 ${__rocm_sdk}/lib)
    # Backward compatible with before ROCm3.7
    if(NOT ROCM_HIPHCC_LIBRARY)
      find_library(ROCM_HIPHCC_LIBRARY hip_hcc ${__rocm_sdk}/lib)
    endif()
    find_library(ROCM_MIOPEN_LIBRARY MIOpen ${__rocm_sdk}/lib)
    find_library(ROCM_ROCBLAS_LIBRARY rocblas ${__rocm_sdk}/lib)
    find_library(ROCM_HSA_LIBRARY hsa-runtime64 ${__rocm_sdk}/lib)

    if(ROCM_HIPHCC_LIBRARY)
      set(ROCM_FOUND TRUE)
    endif()
  endif(__rocm_sdk)
  if(ROCM_FOUND)
    message(STATUS "Found ROCM_INCLUDE_DIRS=" ${ROCM_INCLUDE_DIRS})
    message(STATUS "Found ROCM_HIPHCC_LIBRARY=" ${ROCM_HIPHCC_LIBRARY})
    message(STATUS "Found ROCM_MIOPEN_LIBRARY=" ${ROCM_MIOPEN_LIBRARY})
    message(STATUS "Found ROCM_ROCBLAS_LIBRARY=" ${ROCM_ROCBLAS_LIBRARY})
  endif(ROCM_FOUND)
endmacro(find_rocm)
