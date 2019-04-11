cmake_minimum_required(VERSION 3.2)
project(tsim C CXX)

set(TVM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../)
set(VTA_DIR ${TVM_DIR}/vta)

include_directories("${TVM_DIR}/include")
include_directories("${TVM_DIR}/3rdparty/dlpack/include")
include_directories("${TVM_DIR}/3rdparty/dmlc-core/include")
include_directories("${TVM_DIR}/vta/src/verilator")

set(CMAKE_C_FLAGS "-O2 -Wall -fPIC -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -fvisibility=hidden -std=c++11")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
endif()

# Module rules
include(cmake/modules/tsim.cmake)
include(cmake/modules/driver.cmake)
