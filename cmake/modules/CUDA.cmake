# CUDA Module
find_cuda(${USE_CUDA})

if(CUDA_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
	include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)

if(USE_CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif()
  message(STATUS "Build with CUDA support")
  file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})
  list(APPEND COMPILER_SRCS src/codegen/opt/build_cuda_on.cc)

  list(APPEND TVM_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})

  if(USE_CUDNN)
    message(STATUS "Build with cuDNN support")
    file(GLOB CONTRIB_CUDNN_SRCS src/contrib/cudnn/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUDNN_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIBRARY})
  endif(USE_CUDNN)

  if(USE_CUBLAS)
    message(STATUS "Build with cuBLAS support")
    file(GLOB CONTRIB_CUBLAS_SRCS src/contrib/cublas/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUBLAS_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUBLAS_LIBRARY})
  endif(USE_CUBLAS)

else(USE_CUDA)
  list(APPEND COMPILER_SRCS src/codegen/opt/build_cuda_off.cc)
endif(USE_CUDA)
