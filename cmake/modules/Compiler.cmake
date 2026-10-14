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

add_library(tvm_objs OBJECT ${COMPILER_SRCS})
target_link_libraries(tvm_objs PUBLIC tvm_ffi_header)
if(TARGET tvm_llvm_header)
  target_link_libraries(tvm_objs PUBLIC tvm_llvm_header)
endif()
if(TVM_VISIBILITY_FLAG)
  target_compile_options(tvm_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
endif()
if(TVM_DEBUG_WITH_ABI_CHANGE)
  target_compile_definitions(tvm_objs PRIVATE TVM_DEBUG_WITH_ABI_CHANGE)
endif()
if(MSVC)
  # Compiler objects import runtime symbols from the separate runtime DLL.
  target_compile_definitions(tvm_objs PRIVATE TVM_EXPORTS TVM_FFI_EXPORTS)
endif()

# Compiler library: built from compiler-only sources.
# Links against tvm_runtime (and tvm_ffi_shared transitively). Loaded with
# RTLD_LOCAL so compiler-internal symbols don't leak into the global namespace.
add_library(tvm_compiler SHARED
  $<TARGET_OBJECTS:tvm_objs>
)
target_link_libraries(tvm_compiler PUBLIC tvm_runtime tvm_ffi_shared)
target_include_directories(tvm_compiler PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
set_property(TARGET tvm_compiler APPEND PROPERTY LINK_OPTIONS "${TVM_NO_UNDEFINED_SYMBOLS}")
set_property(TARGET tvm_compiler APPEND PROPERTY LINK_OPTIONS "${TVM_VISIBILITY_FLAG}")

# Work around a GNU ld (binutils) relaxation bug that miscompiles
# R_X86_64_GOTPCRELX relocations inside very large statically-linked archives.
# When the full LLVM static libraries are linked into libtvm_compiler.so, the
# library is large enough that ld can relax an indirect GOT call (LLVM built
# with -fno-plt emits these) into a direct call with an incorrect displacement.
# The call then targets read-only data instead of the intended function and
# crashes at runtime with a SIGSEGV inside llvm::X86Subtarget during code
# generation. Disabling linker relaxation keeps the GOT-indirect sequences and
# avoids the miscompilation; it is harmless when LLVM is linked dynamically.
# See binutils bug ld/25754.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT "${USE_LLVM}" MATCHES ${IS_FALSE_PATTERN})
  set_property(TARGET tvm_compiler APPEND PROPERTY LINK_OPTIONS "-Wl,--no-relax")
  # LLVM's --system-libs reports -lxml2, but TVM calls no libxml2 symbols. The
  # manylinux toolchain does not default to --as-needed, so set it explicitly:
  # the unused libxml2 is then not recorded as a NEEDED dependency, so auditwheel
  # does not vendor it (and its deps) into the wheel.
  set_property(TARGET tvm_compiler APPEND PROPERTY LINK_OPTIONS "-Wl,--as-needed")
endif()

set_target_properties(tvm_compiler PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
if(MSVC)
  foreach(config_type Release RELEASE Debug DEBUG RelWithDebInfo RELWITHDEBINFO MinSizeRel MINSIZEREL)
    set_target_properties(tvm_compiler PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
    )
  endforeach()
endif()

if(USE_THREADS AND NOT BUILD_FOR_HEXAGON)
  target_link_libraries(tvm_compiler PUBLIC Threads::Threads)
endif()
target_link_libraries(tvm_compiler PRIVATE ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})

if(TVM_IS_DEBUG_BUILD)
  target_compile_options(tvm_compiler PRIVATE "${FILE_PREFIX_MAP_FLAG}")
  if(FILE_PREFIX_MAP_SUPPORTED)
    target_compile_options(tvm_compiler PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
    target_compile_options(tvm_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
  endif()
endif()

tvm_ffi_add_apple_dsymutil(tvm_compiler)
tvm_configure_target_library(tvm_compiler)
install(TARGETS tvm_compiler DESTINATION lib${LIB_SUFFIX})
