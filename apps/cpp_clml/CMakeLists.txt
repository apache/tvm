cmake_minimum_required(VERSION 3.13)

project(clml_run VERSION 2.0)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  message( FATAL_ERROR "CMAKE_TOOLCHAIN_FILE Not set, forcing exit. Suggested value: {ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake." )
endif(NOT DEFINED CMAKE_TOOLCHAIN_FILE)

if(NOT DEFINED ANDROID_ABI)
  message( FATAL_ERROR "ANDROID_ABI Not set, forcing exit. Suggested value(s): arm64-v8a (64), armeabi-v7a (32)" )
endif(NOT DEFINED ANDROID_ABI)

if(NOT DEFINED CLML_SDK)
  message( FATAL_ERROR "CLML_SDK Not set, forcing exit." )
endif(NOT DEFINED CLML_SDK)

# CMake/Android variables
set( ANDROID_STL  c++_static CACHE STRING "Target Android STL") # default

# Source variables
set( OPENCL_INCLUDE_DIRS  ${CLML_SDK} CACHE PATH "filepath to OpenCL headers")
set( ANDROID_SOURCE_TREE /path/to/android/au/ CACHE FILEPATH "optional filepath to the Android AU Tree, for building examples using ION Buffers") # tree required to build ION/DMA Buffer samples

#c++ 11 is required
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_FLAGS "-Wall -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#we do not want to pass -fno-exceptions
if(${CMAKE_CXX_FLAGS} MATCHES "-fno-exceptions")
  string(REGEX REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

#we do not want to pass -fno-rtti
if(${CMAKE_CXX_FLAGS} MATCHES "-fno-rtti")
  string(REGEX REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

set(COMMON_SOURCE_FILES
        clml_models.cc
        clml_runner.cc
        clml_runner.h
        main.cc
        ../../3rdparty/cnpy/cnpy.cpp
        )

include_directories(
        src
        ${OPENCL_INCLUDE_DIRS}
        "../../3rdparty/dmlc-core/include"
        "../../3rdparty/cnpy/"
        )

add_executable(clml_run ${COMMON_SOURCE_FILES})
target_link_options(clml_run PRIVATE -Wl,--unresolved-symbols=ignore-in-shared-libs)
target_link_libraries(clml_run ${CLML_SDK}/lib64/libOpenCL.so z)
