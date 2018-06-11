# CUDA Module
find_package(CUDA QUIET)

if(CUDA_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
	include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

if(USE_CUDA)
  find_package(CUDA REQUIRED)
  # Find CUDA doesn't find all the libraries we need, add the extra ones
  find_library(CUDA_CUDA_LIBRARIES cuda
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
  find_library(CUDA_NVRTC_LIBRARIES nvrtc
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
  if(CUDA_CUDA_LIBRARIES)
    set(CUDA_CUDA_LIBRARY ${CUDA_CUDA_LIBRARIES})
  endif()
  message(STATUS "Build with CUDA support")
  file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})
  list(APPEND COMPILER_SRCS src/codegen/opt/build_cuda_on.cc)

  if(MSVC)
    find_library(CUDA_NVRTC_LIB nvrtc
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
    list(APPEND TVM_LINKER_LIBS ${CUDA_NVRTC_LIB})
  else(MSVC)
    find_library(CUDA_NVRTC_LIB nvrtc
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64
      ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    list(APPEND TVM_LINKER_LIBS ${CUDA_NVRTC_LIB})
  endif(MSVC)

  if(USE_CUDNN)
    message(STATUS "Build with cuDNN support")
    file(GLOB CONTRIB_CUDNN_SRCS src/contrib/cudnn/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUDNN_SRCS})
    if(MSVC)
      find_library(CUDA_CUDNN_LIB cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
      list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIB})
    else(MSVC)
      find_library(CUDA_CUDNN_LIB cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)
      list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIB})
    endif(MSVC)
  endif(USE_CUDNN)

  if(USE_CUBLAS)
    message(STATUS "Build with cuBLAS support")
    file(GLOB CONTRIB_CUBLAS_SRCS src/contrib/cublas/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUBLAS_SRCS})
    if(MSVC)
      find_library(CUDA_CUBLAS_LIB cublas
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
      list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUBLAS_LIB})
    else(MSVC)
      find_library(CUDA_CUBLAS_LIB cublas
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)
      list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUBLAS_LIB})
    endif(MSVC)
  endif(USE_CUBLAS)
else(USE_CUDA)
  list(APPEND COMPILER_SRCS src/codegen/opt/build_cuda_off.cc)
endif(USE_CUDA)
