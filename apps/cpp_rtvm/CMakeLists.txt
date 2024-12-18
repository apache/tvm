cmake_policy(SET CMP0069 NEW) # suppress cmake warning about IPO

set(RTVM_SOURCES
  main.cc
  tvm_runner.cc
  ../../3rdparty/cnpy/cnpy.cpp
)
set(TVM_RUNNER_SOURCES
  tvm_runner.cc
  ../../3rdparty/cnpy/cnpy.cpp
)

set(RTVM_LINKER_LIBS "")

if(WIN32)
  file(GLOB ZLIB_SRC
       "../../3rdparty/zlib/*.c"
  )
  list(APPEND RTVM_SOURCES ${ZLIB_SRC})
  list(APPEND TVM_RUNNER_SOURCES ${ZLIB_SRC})
endif()

# Set output to same directory as the other TVM libs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(rtvm ${RTVM_SOURCES})
add_library(tvm_runner_objs OBJECT ${TVM_RUNNER_SOURCES})
add_library(tvm_runner SHARED $<TARGET_OBJECTS:tvm_runner_objs>)

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET rtvm PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(WIN32)
  target_compile_definitions(rtvm PUBLIC -DNOMINMAX)
endif()

if (OS)
   if (OS STREQUAL "Linux")
      set_property(TARGET rtvm PROPERTY LINK_FLAGS -lpthread)
      set_property(TARGET tvm_runner PROPERTY LINK_FLAGS -lpthread)
   endif()
endif()

if(USE_OPENCL)
   if (ANDROID_ABI)
     if(DEFINED ENV{ANDROID_NDK_MAJOR})
       if($ENV{ANDROID_NDK_MAJOR} VERSION_LESS "23")
         set_property(TARGET rtvm PROPERTY LINK_FLAGS -fuse-ld=gold)
         set_property(TARGET tvm_runner PROPERTY LINK_FLAGS -fuse-ld=gold)
       endif()
     endif()
   endif()
endif()

target_include_directories(
  rtvm
  PUBLIC "../../include"
  PUBLIC "../../3rdparty/cnpy"
  PUBLIC "../../3rdparty/zlib"
  PUBLIC DLPACK_PATH
  PUBLIC DMLC_PATH
)

if (BUILD_FOR_ANDROID AND USE_HEXAGON)
  get_hexagon_sdk_property("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}"
    DSPRPC_LIB DSPRPC_LIB_DIRS
  )
  if(DSPRPC_LIB_DIRS)
    link_directories(${DSPRPC_LIB_DIRS})
  else()
    message(WARNING "Could not locate some Hexagon SDK components")
  endif()
  list(APPEND RTVM_LINKER_LIBS cdsprpc log)
endif()

if(BUILD_STATIC_RUNTIME)
  list(APPEND RTVM_LINKER_LIBS -Wl,--whole-archive tvm_runtime -Wl,--no-whole-archive)
else()
  list(APPEND RTVM_LINKER_LIBS tvm_runtime)
endif()

if(NOT WIN32)
  list(APPEND RTVM_LINKER_LIBS z)
endif()

target_link_libraries(rtvm ${RTVM_LINKER_LIBS})

# Build tvm_runner as a exportable lib
target_include_directories(
  tvm_runner_objs
  PUBLIC "../../include"
  PUBLIC "../../3rdparty/cnpy"
  PUBLIC "../../3rdparty/zlib"
  PUBLIC DLPACK_PATH
  PUBLIC DMLC_PATH
)
target_link_libraries(tvm_runner ${RTVM_LINKER_LIBS})
