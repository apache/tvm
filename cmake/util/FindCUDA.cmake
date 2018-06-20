#######################################################
# Enhanced version of find CUDA.
#
# Usage:
#   find_cuda(${USE_CUDA})
#
# - When USE_CUDA=ON, use auto search
# - When USE_CUDA=/path/to/cuda-path, use the cuda path
#
# Provide variables:
#
# - CUDA_FOUND
# - CUDA_INCLUDE_DIRS
# - CUDA_TOOLKIT_ROOT_DIR
# - CUDA_CUDA_LIBRARY
# - CUDA_CUDART_LIBRARY
# - CUDA_NVRTC_LIBRARY
# - CUDA_CUDNN_LIBRARY
# - CUDA_CUBLAS_LIBRARY
#
macro(find_cuda use_cuda)
  set(__use_cuda ${use_cuda})
  if(__use_cuda STREQUAL "ON")
    find_package(CUDA QUIET)
  elseif(IS_DIRECTORY ${__use_cuda})
    set(CUDA_TOOLKIT_ROOT_DIR ${__use_cuda})
    message(STATUS "Custom CUDA_PATH=" ${CUDA_TOOLKIT_ROOT_DIR})
    set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
    set(CUDA_FOUND TRUE)
    if(MSVC)
      find_library(CUDA_CUDAT_LIBRARY cudart
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    else(MSVC)
      find_library(CUDA_CUDAT_LIBRARY cudart
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    endif(MSVC)
  endif()

  # additional libraries
  if(CUDA_FOUND)
    if(MSVC)
      find_library(CUDA_NVRTC_LIBRARY cuda
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
      find_library(CUDA_NVRTC_LIBRARY nvrtc
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
      find_library(CUDA_CUDNN_LIBRARY cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
      find_library(CUDA_CUBLAS_LIBRARY cublas
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
    else(MSVC)
      find_library(_CUDA_CUDA_LIBRARY cuda
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
      if(_CUDA_CUDA_LIBRARY)
        set(CUDA_CUDA_LIBRARY ${_CUDA_CUDA_LIBRARY})
      endif()
      find_library(CUDA_NVRTC_LIBRARY nvrtc
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
      find_library(CUDA_CUDNN_LIBRARY cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)
      find_library(CUDA_CUBLAS_LIBRARY cublas
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    endif(MSVC)
  endif(CUDA_FOUND)
endmacro(find_cuda)
