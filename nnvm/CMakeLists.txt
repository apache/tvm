cmake_minimum_required(VERSION 2.8.7)

project(nnvm C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/build/private/local_config.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/build/private/local_config.cmake)
endif()

include(cmake/Utils.cmake)

# include path
include_directories(BEFORE "include")

set(nnvm_LINKER_LIBS "")

add_definitions(-DNNVM_EXPORTS)

# compile
if(MSVC)
  add_definitions(-DDMLC_USE_CXX11)
  add_definitions(-DDMLC_STRICT_CXX11)
  foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
      string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MD")
  endforeach(flag_var)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++0x"    SUPPORT_CXX0X)
  check_cxx_compiler_flag("-msse2"        SUPPORT_MSSE2)
  check_cxx_compiler_flag("-openmp"       SUPPORT_OPENMP)
  set(CMAKE_C_FLAGS "-O3 -Wall -msse2 -Wno-unknown-pragmas -std=c++0x -fPIC")
  if(SUPPORT_OPENMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
  endif()
  set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
endif(MSVC)

mxnet_source_group("Include\\c_api"   GLOB "src/c_api/*.h")
mxnet_source_group("Include\\core"   GLOB "src/core/*.h")
mxnet_source_group("Include\\pass"   GLOB "src/pass/*.h")
mxnet_source_group("Include\\nnvm"   GLOB "include/nnvm/*.h")
mxnet_source_group("Include\\dmlc"   GLOB "include/dmlc/*.h")

mxnet_source_group("Source"   GLOB "src/*.cc")
mxnet_source_group("Source\\c_api"   GLOB "src/c_api/*.cc")
mxnet_source_group("Source\\core"   GLOB "src/core/*.cc")
mxnet_source_group("Source\\pass"   GLOB "src/pass/*.cc")


FILE(GLOB_RECURSE SOURCE "src/*.cc" "src/*.h" "include/*.h")

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dmlc-core/CMakeLists.txt)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dmlc-core/include)
elseif(DMLC_CORE_PATH)
  include_directories(${DMLC_CORE_PATH}/include)
endif()

add_library(nnvm SHARED ${SOURCE})
target_link_libraries(nnvm ${nnvm_LINKER_LIBS})

if(INSTALL_INCLUDE_DIR)
  add_custom_command(TARGET nnvm POST_BUILD
    COMMAND COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/include ${INSTALL_INCLUDE_DIR}/
    )
endif()

# ---[ Linter target
if(MSVC)
  find_package(PythonInterp 2)
  set(PYTHON2_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Path to the python 2.x executable")
  find_package(PythonInterp 3)
  set(PYTHON3_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Path to the python 3.x executable")
endif()
set(LINT_DIRS include src scripts)
add_custom_target(nnvm_lint COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DPYTHON2_EXECUTABLE=${PYTHON2_EXECUTABLE} -DPYTHON3_EXECUTABLE=${PYTHON3_EXECUTABLE} -DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR} -DLINT_DIRS=${LINT_DIRS} -DPROJECT_NAME=dmlc -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)
