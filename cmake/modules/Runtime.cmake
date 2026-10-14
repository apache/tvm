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

# Core runtime objects are shared by the dynamic and static libraries.
add_library(tvm_runtime_objs OBJECT ${RUNTIME_SRCS})
target_link_libraries(tvm_runtime_objs PUBLIC tvm_ffi_header)
if(TVM_VISIBILITY_FLAG)
  target_compile_options(tvm_runtime_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
endif()
if(TVM_DEBUG_WITH_ABI_CHANGE)
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_DEBUG_WITH_ABI_CHANGE)
endif()
if(MSVC)
  # Runtime objects export only runtime/FFI symbols, never compiler symbols.
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_RUNTIME_EXPORTS TVM_FFI_EXPORTS)
endif()

# Loaded with RTLD_GLOBAL so downstream runtime modules can resolve its symbols.
add_library(tvm_runtime SHARED
  $<TARGET_OBJECTS:tvm_runtime_objs>
  ${TVM_RUNTIME_EXT_OBJS}
)
target_link_libraries(tvm_runtime PUBLIC tvm_ffi_shared)
set_property(TARGET tvm_runtime APPEND PROPERTY LINK_OPTIONS "${TVM_NO_UNDEFINED_SYMBOLS}")
set_property(TARGET tvm_runtime APPEND PROPERTY LINK_OPTIONS "${TVM_VISIBILITY_FLAG}")

# Static consumers must retain registration objects from both archives, with
# tvm_ffi_static before tvm_runtime_static so FFI type attributes initialize first:
#   -Wl,--whole-archive tvm_ffi_static tvm_runtime_static -Wl,--no-whole-archive
# Use the corresponding archive-retention flags on other platforms.
add_library(tvm_runtime_static STATIC
  $<TARGET_OBJECTS:tvm_runtime_objs>
  ${TVM_RUNTIME_EXT_OBJS}
)
target_link_libraries(tvm_runtime_static PUBLIC tvm_ffi_static)
if(MSVC)
  # Static consumers reference ordinary symbols rather than DLL import stubs.
  target_compile_definitions(tvm_runtime_static INTERFACE
    TVM_RUNTIME_DLL= TVM_RUNTIME_DLL_EXPORT= TVM_FFI_DLL= TVM_FFI_DLL_EXPORT=)
endif()

foreach(runtime_target tvm_runtime tvm_runtime_static)
  target_include_directories(${runtime_target} PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_link_libraries(${runtime_target} PRIVATE ${TVM_RUNTIME_LINKER_LIBS})
  if(USE_THREADS AND NOT BUILD_FOR_HEXAGON)
    target_link_libraries(${runtime_target} PUBLIC Threads::Threads)
  endif()
endforeach()

set_target_properties(tvm_runtime tvm_runtime_static PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
if(MSVC)
  foreach(config_type Release RELEASE Debug DEBUG RelWithDebInfo RELWITHDEBINFO MinSizeRel MINSIZEREL)
    set_target_properties(tvm_runtime tvm_runtime_static PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
    )
  endforeach()
endif()

if(TVM_IS_DEBUG_BUILD AND FILE_PREFIX_MAP_SUPPORTED)
  target_compile_options(tvm_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
  target_compile_options(tvm_runtime_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
endif()

tvm_ffi_add_apple_dsymutil(tvm_runtime)
tvm_configure_target_library(tvm_runtime)
add_custom_target(runtime DEPENDS tvm_runtime tvm_runtime_static)
install(TARGETS tvm_runtime DESTINATION lib${LIB_SUFFIX})
# Keep wheel contents unchanged; the archive is available to native consumers.
if(NOT TVM_BUILD_PYTHON_MODULE)
  install(TARGETS tvm_runtime_static DESTINATION lib${LIB_SUFFIX})
endif()
