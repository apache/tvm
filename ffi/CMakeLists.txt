# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
cmake_minimum_required(VERSION 3.18)

project(
  tvm_ffi
  LANGUAGES CXX C
)

option(TVM_FFI_USE_LIBBACKTRACE "Enable libbacktrace" ON)
option(TVM_FFI_USE_EXTRA_CXX_API "Enable extra CXX API in shared lib" ON)
option(TVM_FFI_BACKTRACE_ON_SEGFAULT "Set signal handler to print traceback on segfault" ON)

if (TVM_FFI_USE_LIBBACKTRACE)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/Utils/AddLibbacktrace.cmake)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/Utils/Library.cmake)


########## Target: `tvm_ffi_header` ##########

# they can be used in cases where user do not want to link into the library
# in cases like deferred linking
add_library(tvm_ffi_header INTERFACE)
target_compile_features(tvm_ffi_header INTERFACE cxx_std_17)
target_include_directories(
  tvm_ffi_header INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(
  tvm_ffi_header INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include>
  $<INSTALL_INTERFACE:include>
)

########## Target: `tvm_ffi_objs` ##########

set(tvm_ffi_objs_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/traceback.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/traceback_win.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/object.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/error.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/function.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/tensor.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/dtype.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/container.cc"
)

set(tvm_ffi_extra_objs_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/structural_equal.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/structural_hash.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/json_parser.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/json_writer.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/serialization.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/reflection_extra.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/module.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/library_module.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/library_module_system_lib.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/library_module_dynamic_lib.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/stream_context.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/env_c_api.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/extra/testing.cc"
)
if (TVM_FFI_USE_EXTRA_CXX_API)
  list(APPEND tvm_ffi_objs_sources ${tvm_ffi_extra_objs_sources})
endif()

add_library(tvm_ffi_objs OBJECT ${tvm_ffi_objs_sources})
target_compile_features(tvm_ffi_objs PRIVATE cxx_std_17)

set_target_properties(
  tvm_ffi_objs PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  PREFIX "lib"
)

# add the include path as public so they are visible to downstreams
target_link_libraries(tvm_ffi_objs PUBLIC tvm_ffi_header)

if (TVM_FFI_USE_LIBBACKTRACE)
  message(STATUS "Setting C++ macro TVM_FFI_USE_LIBBACKTRACE - 1")
  target_compile_definitions(tvm_ffi_objs PRIVATE TVM_FFI_USE_LIBBACKTRACE=1)
else()
  message(STATUS "Setting C++ macro TVM_FFI_USE_LIBBACKTRACE - 0")
  target_compile_definitions(tvm_ffi_objs PRIVATE TVM_FFI_USE_LIBBACKTRACE=0)
endif()

if (TVM_FFI_BACKTRACE_ON_SEGFAULT)
  message(STATUS "Setting C++ macro TVM_FFI_BACKTRACE_ON_SEGFAULT - 1")
  target_compile_definitions(tvm_ffi_objs PRIVATE TVM_FFI_BACKTRACE_ON_SEGFAULT=1)
else()
  message(STATUS "Setting C++ macro TVM_FFI_BACKTRACE_ON_SEGFAULT - 0")
  target_compile_definitions(tvm_ffi_objs PRIVATE TVM_FFI_BACKTRACE_ON_SEGFAULT=0)
endif()

tvm_ffi_add_msvc_flags(tvm_ffi_objs)
tvm_ffi_add_target_from_obj(tvm_ffi tvm_ffi_objs)

if (TARGET libbacktrace)
  target_link_libraries(tvm_ffi_objs PRIVATE libbacktrace)
  target_link_libraries(tvm_ffi_shared PRIVATE libbacktrace)
  target_link_libraries(tvm_ffi_static PRIVATE libbacktrace)
endif ()

if (MSVC)
  target_link_libraries(tvm_ffi_objs PRIVATE DbgHelp.lib)
  target_link_libraries(tvm_ffi_shared PRIVATE DbgHelp.lib)
  target_link_libraries(tvm_ffi_static PRIVATE DbgHelp.lib)
  # produce pdb file
  target_link_options(tvm_ffi_shared PRIVATE /DEBUG)
endif ()

# expose the headers as public dependencies
target_link_libraries(tvm_ffi_objs PUBLIC tvm_ffi_header)
target_link_libraries(tvm_ffi_shared PUBLIC tvm_ffi_header)
target_link_libraries(tvm_ffi_static PUBLIC tvm_ffi_header)

#----------------------------------------------------------------------------
# The following code section only is triggered when the project is the root
# and will be skipped when the project is a subproject.
#----------------------------------------------------------------------------
if (NOT ${PROJECT_NAME} STREQUAL ${CMAKE_PROJECT_NAME})
  return()
endif()

option(TVM_FFI_ATTACH_DEBUG_SYMBOLS "Attach debug symbols even in release mode" OFF)
option(TVM_FFI_BUILD_TESTS "Adding test targets." OFF)

if (TVM_FFI_ATTACH_DEBUG_SYMBOLS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tvm_ffi_objs PRIVATE -g1)
  endif()
endif()

include(cmake/Utils/CxxWarning.cmake)
include(cmake/Utils/Sanitizer.cmake)

# remap the file name to the source directory so we can see the
# exact file name in traceback relative to the project source root
tvm_ffi_add_prefix_map(tvm_ffi_objs ${CMAKE_SOURCE_DIR})

########## Adding cpp tests ##########

# logics below are only executed when the project is the root project.
# but not when the project is a subproject.
if (TVM_FFI_BUILD_TESTS)
  enable_testing()
  message(STATUS "Enable Testing")
  include(cmake/Utils/AddGoogleTest.cmake)
  add_subdirectory(tests/cpp/)
  tvm_ffi_add_cxx_warning(tvm_ffi_objs)
endif()

########## Adding python module ##########
option(TVM_FFI_BUILD_PYTHON_MODULE "Adding python module." OFF)

if (TVM_FFI_BUILD_PYTHON_MODULE)
  # Helper function to build the cython module
  message(STATUS "Building cython module..")
  find_package(
    Python COMPONENTS Interpreter Development.Module Development.SABIModule
    REQUIRED)
  set(core_cpp ${CMAKE_CURRENT_BINARY_DIR}/core.cpp)
  set(core_pyx ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_ffi/cython/core.pyx)
  set(cython_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_ffi/cython/core.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_ffi/cython/base.pxi
    ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_ffi/cython/device.pxi
    ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_ffi/cython/dtype.pxi
    ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_ffi/cython/error.pxi
    ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_ffi/cython/function.pxi
    ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_ffi/cython/tensor.pxi
    ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_ffi/cython/object.pxi
    ${CMAKE_CURRENT_SOURCE_DIR}/python/tvm_ffi/cython/string.pxi
  )
  # set working directory to source so we can see the exact file name in traceback
  # relatived to the project source root
  add_custom_command(
      OUTPUT ${core_cpp}
      COMMAND ${Python_EXECUTABLE} -m cython --cplus ${core_pyx} -o ${core_cpp}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Transpiling ${core_pyx} to ${core_cpp}"
      DEPENDS ${cython_sources}
      VERBATIM
  )
  if(Python_VERSION VERSION_GREATER_EQUAL "3.12")
    # >= Python3.12, use Use_SABI version
    Python_add_library(tvm_ffi_cython MODULE "${core_cpp}" USE_SABI 3.12)
    set_target_properties(tvm_ffi_cython PROPERTIES OUTPUT_NAME "core")
    if(NOT WIN32)
      set_target_properties(tvm_ffi_cython PROPERTIES SUFFIX ".abi3.so")
    endif()
  else()
    # before Python3.12, use WITH_SOABI version
    Python_add_library(tvm_ffi_cython MODULE "${core_cpp}" WITH_SOABI)
    set_target_properties(tvm_ffi_cython PROPERTIES OUTPUT_NAME "core")
  endif()
  target_compile_features(tvm_ffi_cython PRIVATE cxx_std_17)
  target_link_libraries(tvm_ffi_cython PRIVATE tvm_ffi_header)
  target_link_libraries(tvm_ffi_cython PRIVATE tvm_ffi_shared)
  # Set RPATH for tvm_ffi_cython to find tvm_ffi_shared.so relatively
  if(APPLE)
    # macOS uses @loader_path
    set_target_properties(tvm_ffi_cython PROPERTIES INSTALL_RPATH "@loader_path/lib")
  elseif(LINUX)
      # Linux uses $ORIGIN
    set_target_properties(tvm_ffi_cython PROPERTIES INSTALL_RPATH "\$ORIGIN/lib")
  endif()
  install(TARGETS tvm_ffi_cython DESTINATION .)

  ########## Installing the source ##########
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include DESTINATION 3rdparty/dlpack/include
  )
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libbacktrace DESTINATION 3rdparty/libbacktrace
    PATTERN ".git" EXCLUDE
    PATTERN ".git*" EXCLUDE
    PATTERN "*.tmp" EXCLUDE
  )
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/ DESTINATION src/ffi/)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Utils/ DESTINATION cmake/Utils)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt DESTINATION .)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tvm_ffi-config.cmake DESTINATION cmake)
endif()

########## Install the related for normal cmake library ##########

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/tvm/ffi/ DESTINATION include/tvm/ffi/)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include/ DESTINATION include/)
install(TARGETS tvm_ffi_shared  DESTINATION lib)
# ship additional dSYM files for debugging symbols on if available
if (APPLE)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/ DESTINATION lib FILES_MATCHING PATTERN "*.dSYM")
endif()

if (NOT TVM_FFI_BUILD_PYTHON_MODULE)
  # when building wheel, we do not ship static as we already ships source and dll
  install(TARGETS tvm_ffi_static DESTINATION lib)
endif()
