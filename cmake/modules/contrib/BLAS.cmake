# Plugin rules for cblas
file(GLOB CBLAS_CONTRIB_SRC src/contrib/cblas/*.cc)

if(USE_BLAS STREQUAL "openblas")
  find_library(BLAS_LIBRARY openblas)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS ${CBLAS_CONTRIB_SRC})
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(USE_BLAS STREQUAL "mkl")
  if(IS_DIRECTORY ${MKL_PATH})
    find_library(BLAS_LIBRARY mkl_rt ${MKL_PATH}/lib/intel64)
    include_directories(${MKL_PATH}/include)
  else()
    find_library(BLAS_LIBRARY mkl_rt /opt/intel/mkl/lib/intel64)
    include_directories(/opt/intel/mkl/include)
  endif()
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS ${CBLAS_CONTRIB_SRC})
  add_definitions(-DUSE_MKL_BLAS=1)
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(USE_BLAS STREQUAL "atlas" OR USE_BLAS STREQUAL "blas")
  find_library(BLAS_LIBRARY cblas)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS ${CBLAS_CONTRIB_SRC})
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(USE_BLAS STREQUAL "apple")
  find_library(BLAS_LIBRARY Accelerate)
  include_directories(${BLAS_LIBRARY}/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${BLAS_LIBRARY})
  list(APPEND RUNTIME_SRCS ${CBLAS_CONTRIB_SRC})
  message(STATUS "Use BLAS library " ${BLAS_LIBRARY})
elseif(USE_BLAS STREQUAL "none")
  # pass
else()
  message(FATAL_ERROR "Invalid option: USE_BLAS=" ${USE_BLAS})
endif()
