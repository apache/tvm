#######################################################
# Enhanced version of find rocm.
#
# Usage:
#   find_rocm(${USE_ROCM})
#
# - When USE_VULKAN=ON, use auto search
# - When USE_VULKAN=/path/to/vulkan-sdk-path, use the sdk
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
    find_library(ROCM_HIPHCC_LIBRARY hip_hcc ${__rocm_sdk}/lib)
    find_library(ROCM_MIOPEN_LIBRARY MIOpen ${__rocm_sdk}/lib)
    find_library(ROCM_ROCBLAS_LIBRARY rocblas ${__rocm_sdk}/lib)
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
