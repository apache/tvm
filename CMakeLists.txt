cmake_minimum_required(VERSION 3.5)
project(tvm)

include(cmake/Util.cmake)

option(USE_OPENCL "Build with OpenCL" OFF)
option(USE_CUDA "Build with CUDA" OFF)
option(USE_LLVM "Build with LLVM" OFF)
option(USE_RTTI "Build with RTTI" OFF)

# include path
include_directories("include")
include_directories("HalideIR/src")
set(TVM_LINKER_LIBS "")
set(TVM_RUNTIME_LINKER_LIBS "")

# compile
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-DTVM_EXPORTS)
  foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MD")
  endforeach(flag_var)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
  check_cxx_compiler_flag("-msse2"        SUPPORT_MSSE2)
  set(CMAKE_C_FLAGS "-O3 -fno-rtti -Wall -std=c++11 -fPIC")
  if(SUPPORT_OPENMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
  endif()
  set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
endif(MSVC)

tvm_source_group("Include\\tvm"   GLOB "include/tvm/*.h")
tvm_source_group("Include\\tvm\\runtime"   GLOB "include/tvm/runtime/*.h")
tvm_source_group("Source\\lang"   GLOB "src/lang/*.cc")
tvm_source_group("Source\\api"   GLOB "src/api/*.cc")
tvm_source_group("Source\\arithmetic"   GLOB "src/arithmetic/*.cc")
tvm_source_group("Source\\schedule"   GLOB "src/schedule/*.cc")
tvm_source_group("Source\\codegen"   GLOB "src/codegen/*.cc")
tvm_source_group("Source\\codegen\\llvm"   GLOB "src/codegen/llvm/*.cc")
tvm_source_group("Source\\pass"   GLOB "src/pass/*.cc")
tvm_source_group("Source\\runtime"   GLOB "src/runtime/*.cc")
tvm_source_group("Source\\runtime\\cuda"   GLOB "src/runtime/cuda/*.cc")
tvm_source_group("Source\\runtime\\opencl"   GLOB "src/runtime/opencl/*.cc")

file(GLOB COMPILER_SRCS
    src/api/*.cc
    src/arithmetic/*.cc
    src/codegen/*.cc
    src/stack_vm/*.cc
    src/lang/*.cc
    src/pass/*.cc
    src/schedule/*.cc
)
file(GLOB_RECURSE HALIDEIR_SRCS HalideIR/src/*.cpp)
list(APPEND COMPILER_SRCS ${HALIDEIR_SRCS})
file(GLOB RUNTIME_SRCS src/runtime/*.cc)
file(GLOB COMPILER_LLVM_SRCS src/codegen/llvm/*.cc)
file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
file(GLOB RUNTIME_OPENCL_SRCS src/runtime/opencl/*.cc)

if(USE_CUDA)
  list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})
else(USE_CUDA)
  add_definitions(-DTVM_CUDA_RUNTIME=0)
endif(USE_CUDA)

if(USE_OPENCL)
  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENCL_SRCS})
else(USE_OPENCL)
  add_definitions(-DTVM_OPENCL_RUNTIME=0)
endif(USE_OPENCL)

if(USE_LLVM)
  add_definitions(-DTVM_LLVM_VERSION=40)
  list(APPEND COMPILER_SRCS ${COMPILER_LLVM_SRCS})
endif(USE_LLVM)

if(NOT USE_RTTI)
  add_definitions(-DDMLC_ENABLE_RTTI=0)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dmlc-core/CMakeLists.txt)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dmlc-core/include)
elseif(DMLC_CORE_PATH)
  include_directories(${DMLC_CORE_PATH}/include)
endif()

# Set library output directories
if(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif()


add_library(libtvm SHARED ${COMPILER_SRCS} ${RUNTIME_SRCS})
add_library(libtvm_runtime SHARED ${RUNTIME_SRCS})

target_link_libraries(libtvm ${TVM_LINKER_LIBS})
target_link_libraries(libtvm_runtime  ${TVM_RUNTIME_LINKER_LIBS})
