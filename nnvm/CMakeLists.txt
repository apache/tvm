cmake_minimum_required(VERSION 2.8.7)

project(nnvm C CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/build/private/local_config.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/build/private/local_config.cmake)
endif()

include(cmake/Utils.cmake)
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# include path
include_directories(BEFORE "include")
include_directories("tvm/include")
include_directories("tvm/dlpack/include")
include_directories("tvm/HalideIR/src")
include_directories("tvm/topi/include")

set(NNVM_LINKER_LIBS "")
set(NNVM_COMPILER_LINKER_LIBS "")
add_definitions(-DNNVM_EXPORTS)

# Build a shared lib (libnnvm.so) by default
option(BUILD_SHARED_NNVM "Build a shared nnvm lib" ON)
option(BUILD_STATIC_NNVM "Build a static nnvm lib" OFF)

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
  find_library(TVM_LIB tvm
			   HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../tvm/build/Release
			   HINTS ${CMAKE_CURRENT_SOURCE_DIR}/tvm/build/Release)
  message(STATUS "Build with TVM libary: " ${TVM_LIB})
  list(APPEND NNVM_COMPILER_LINKER_LIBS ${TVM_LIB})
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
  set(CMAKE_C_FLAGS "-O3 -Wall -std=c++11 -fPIC")
  set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
endif(MSVC)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
endif()


mxnet_source_group("Include\\c_api"   GLOB "src/c_api/*.h")
mxnet_source_group("Include\\core"   GLOB "src/core/*.h")
mxnet_source_group("Include\\pass"   GLOB "src/pass/*.h")
mxnet_source_group("Include\\nnvm"   GLOB "include/nnvm/*.h")
mxnet_source_group("Include\\dmlc"   GLOB "include/dmlc/*.h")

mxnet_source_group("Source"   GLOB "src/*.cc")
mxnet_source_group("Source\\c_api"   GLOB "src/c_api/*.cc")
mxnet_source_group("Source\\core"   GLOB "src/core/*.cc")
mxnet_source_group("Source\\pass"   GLOB "src/pass/*.cc")


file(GLOB_RECURSE SOURCE
   src/c_api/*.cc
   src/core/*.cc
   src/pass/*.cc)

file(GLOB_RECURSE COMPILER_SRCS
    src/compiler/*.cc
    src/top/*.cc
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dmlc-core/CMakeLists.txt)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dmlc-core/include)
elseif(DMLC_CORE_PATH)
  include_directories(${DMLC_CORE_PATH}/include)
endif()

if(BUILD_SHARED_NNVM)
  add_library(nnvm SHARED ${SOURCE})
  target_link_libraries(nnvm ${nnvm_LINKER_LIBS})
endif()
if(BUILD_STATIC_NNVM)
  add_library(nnvm_static STATIC ${SOURCE})
  target_link_libraries(nnvm_static ${nnvm_LINKER_LIBS})
  set_target_properties(nnvm_static PROPERTIES OUTPUT_NAME "nnvm")
endif()

add_library(nnvm_compiler SHARED ${COMPILER_SRCS} ${SOURCE})
target_link_libraries(nnvm_compiler ${NNVM_COMPILER_LINKER_LIBS} ${NNVM_LINKER_LIBS})

if(INSTALL_INCLUDE_DIR)
  add_custom_command(TARGET nnvm POST_BUILD
    COMMAND COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/include ${INSTALL_INCLUDE_DIR}/
    )
endif()

# ---[ Install lib, header and docs
if(BUILD_SHARED_NNVM)
  if(WIN32)
    install(TARGETS nnvm RUNTIME DESTINATION bin)
    install(TARGETS nnvm ARCHIVE DESTINATION lib)
    install(TARGETS nnvm_compiler RUNTIME DESTINATION bin)
    install(TARGETS nnvm_compiler ARCHIVE DESTINATION lib)
  else()
    install(TARGETS nnvm LIBRARY DESTINATION lib)
    install(TARGETS nnvm_compiler LIBRARY DESTINATION lib)
  endif()
endif()
if(BUILD_STATIC_NNVM)
  install(TARGETS nnvm_static ARCHIVE DESTINATION lib)
endif()
install(DIRECTORY include DESTINATION .)
install(DIRECTORY docs DESTINATION .)

# ---[ Linter target
if(MSVC)
  find_package(PythonInterp 2)
  set(PYTHON2_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Path to the python 2.x executable")
  find_package(PythonInterp 3)
  set(PYTHON3_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Path to the python 3.x executable")
endif()
set(LINT_DIRS include src scripts)
add_custom_target(nnvm_lint COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DPYTHON2_EXECUTABLE=${PYTHON2_EXECUTABLE} -DPYTHON3_EXECUTABLE=${PYTHON3_EXECUTABLE} -DPROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR} -DLINT_DIRS=${LINT_DIRS} -DPROJECT_NAME=dmlc -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)
