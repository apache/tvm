# ROCM Module
find_rocm(${USE_ROCM})

if(ROCM_FOUND)
  # always set the includedir
  # avoid global retrigger of cmake
  include_directories(${ROCM_INCLUDE_DIRS})
  add_definitions(-D__HIP_PLATFORM_HCC__=1)
endif(ROCM_FOUND)


if(USE_ROCM)
  if(NOT ROCM_FOUND)
    message(FATAL_ERROR "Cannot find ROCM, USE_ROCM=" ${USE_ROCM})
  endif()
  message(STATUS "Build with ROCM support")
  file(GLOB RUNTIME_ROCM_SRCS src/runtime/rocm/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_ROCM_SRCS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${ROCM_HIPHCC_LIBRARY})

  if(USE_MIOPEN)
    message(STATUS "Build with MIOpen support")
    file(GLOB MIOPEN_CONTRIB_SRCS src/contrib/miopen/*.cc)
    list(APPEND RUNTIME_SRCS ${MIOPEN_CONTRIB_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${ROCM_MIOPEN_LIBRARY})
  endif(USE_MIOPEN)

  if(USE_ROCBLAS)
    message(STATUS "Build with RocBLAS support")
    file(GLOB ROCBLAS_CONTRIB_SRCS src/contrib/rocblas/*.cc)
    list(APPEND RUNTIME_SRCS ${ROCBLAS_CONTRIB_SRCS})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${ROCM_ROCBLAS_LIBRARY})
  endif(USE_ROCBLAS)
else(USE_ROCM)
  list(APPEND COMPILER_SRCS src/codegen/opt/build_rocm_off.cc)
endif(USE_ROCM)
