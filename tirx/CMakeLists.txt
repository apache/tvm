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
project(tvm_tirx C CXX)
include(GNUInstallDirs)
if(TVM_BUILD_PYTHON_MODULE)
  set(CMAKE_INSTALL_LIBDIR lib)
endif()

# The extracted package has an independent version; parent SCM tags do not apply.
set(TVM_TIRX_VERSION "0.1.0a0")
set(TVM_VERSION "${TVM_TIRX_VERSION}")
add_compile_definitions(TVM_VERSION="${TVM_VERSION}")

# Utility functions
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/Utils.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/Summary.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/Linker.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/Library.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/FindCUDA.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/FindOpenCL.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/FindVulkan.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/FindLLVM.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/FindROCM.cmake")

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
tvm_option(USE_CUDA "Build with CUDA" OFF)
tvm_option(USE_OPENCL "Build with OpenCL" OFF)
tvm_option(USE_OPENCL_ENABLE_HOST_PTR "Enable OpenCL memory object access to host" OFF)
tvm_option(USE_OPENCL_GTEST "Path to OpenCL specific gtest version for runtime cpp tests." /path/to/opencl/gtest)
tvm_option(USE_VULKAN "Build with Vulkan" OFF)


# Whether to use spirv-tools and SPIRV-Headers from Khronos GitHub or GitLab.
#
# Possible values:
# - OFF: not to use
# - /path/to/install: path to your khronis spirv-tools and SPIRV-Headers installation directory
#
tvm_option(USE_KHRONOS_SPIRV "Whether to use spirv-tools and SPIRV-Headers from Khronos GitHub or GitLab" OFF)
tvm_option(USE_SPIRV_KHR_INTEGER_DOT_PRODUCT "whether enable SPIRV_KHR_DOT_PRODUCT" OFF)
tvm_option(USE_METAL "Build with Metal" OFF)
tvm_option(USE_ROCM "Build with ROCM" OFF)
tvm_option(ROCM_PATH "The path to rocm" /opt/rocm)
tvm_option(USE_HEXAGON_QHL "Enable Hexagon QHL compiler lowering (requires external runtime wrappers)" OFF)

tvm_option(USE_THREADS "Build with thread support" ON)
tvm_option(USE_LLVM "Build with LLVM, can be set to specific llvm-config path" OFF)
tvm_option(USE_OPENMP "Build with OpenMP thread pool implementation" OFF)
tvm_option(TVM_LOG_BEFORE_THROW "Whether log before throw, for debugging purposes" OFF)
tvm_option(USE_RTTI "Build with RTTI" ON)
tvm_option(USE_MSVC_MT "Build with MT" OFF)
tvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)
tvm_option(HIDE_PRIVATE_SYMBOLS "Compile with -fvisibility=hidden." ON)
tvm_option(INDEX_DEFAULT_I64 "Defaults the index datatype to int64" ON)
tvm_option(USE_GTEST "Use GoogleTest for C++ sanity tests" AUTO)
tvm_option(USE_CUSTOM_LOGGING "Use user-defined custom logging, tvm::runtime::detail::LogFatalImpl and tvm::runtime::detail::LogMessageImpl must be implemented" OFF)
tvm_option(USE_ALTERNATIVE_LINKER "Use 'mold' or 'lld' if found when invoking compiler to link artifact" AUTO)
tvm_option(USE_CCACHE "Use ccache if found when invoking compiler" AUTO)

# 3rdparty libraries
tvm_option(COMPILER_RT_PATH "Path to COMPILER-RT" "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/compiler-rt")
# Contrib library options
tvm_option(USE_Z3 "Build with Z3 SMT solver support" AUTO)
tvm_option(USE_NVTX "Build with NVTX" OFF)
tvm_option(SUMMARIZE "Print CMake option summary after configuring" OFF)

# Python package options
tvm_option(TVM_BUILD_PYTHON_MODULE "Build Python module with scikit-build-core" OFF)

# include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories("include")
include_directories(SYSTEM ${COMPILER_RT_PATH})

# initial variables
set(TVM_LINKER_LIBS "")
set(TVM_RUNTIME_LINKER_LIBS "")
set(TVM_RUNTIME_BACKEND_LIBS "")
# Use one external FFI registry for both the core and its parent consumers.
find_package(Python COMPONENTS Interpreter REQUIRED)
if(NOT tvm_ffi_DIR)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
    OUTPUT_VARIABLE tvm_ffi_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
endif()
if(NOT TARGET tvm_ffi::shared)
  find_package(tvm_ffi CONFIG REQUIRED)
  foreach(_ffi_target header shared static)
    if(TARGET tvm_ffi::${_ffi_target})
      set_target_properties(tvm_ffi::${_ffi_target} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
  endforeach()
endif()
if(NOT TARGET tvm_ffi_header)
  add_library(tvm_ffi_header ALIAS tvm_ffi::header)
endif()

# Check if this is being run on its own or as a subdirectory for another project
# If we update to CMake 2.21+, we can use PROJECT_IS_TOP_LEVEL instead
get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)

if(NOT IS_SUBPROJECT AND NOT DEFINED "${CMAKE_EXPORT_COMPILE_COMMANDS}")
  # If not set manually, change the default to ON
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(TVM_LOG_BEFORE_THROW)
  # log error before throw as
  # when system have issues with stack trace
  add_definitions(-DTVM_LOG_BEFORE_THROW=1)
endif()

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
  add_definitions(-DNOMINMAX)
  # regeneration does not work well with msbuild custom rules.
  set(CMAKE_SUPPRESS_REGENERATION ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  add_compile_options(/bigobj)

  # Use standard-conforming two-phase name resolution for templates.
  # This minimizes the differences between g++/clang builds on Linux,
  # and MSVC builds on Windows.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")

  # MSVC already errors on undefined symbols, no additional flag needed.
  set(TVM_NO_UNDEFINED_SYMBOLS "")

  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
    # Static linking. CMake behavior changed in 3.15 making this necessary.
    add_compile_options(/MT)
  endif()
  # Disable common MSVC warnings
  # Integer conversion warnings(e.g. int64 to int)
  add_compile_options(/wd4244)
  add_compile_options(/wd4267)
  # Signed unsigned constant comparison
  add_compile_options(/wd4018)
  # Aligned alloc may not met(need c++17)
  add_compile_options(/wd4316)
  # unreferenced local variables(usually in exception catch)
  add_compile_options(/wd4101)
  # always inline keyword not necessary
  add_compile_options(/wd4180)
  # DLL interface warning in c++
  add_compile_options(/wd4251)
  # destructor was implicitly defined as deleted
  add_compile_options(/wd4624)
  # unary minus operator applied to unsigned type, result still unsigned
  add_compile_options(/wd4146)
  # 'inline': used more than once
  add_compile_options(/wd4141)
  # unknown pragma
  add_compile_options(/wd4068)
else(MSVC)
  set(WARNING_FLAG -Wall)
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 -g ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  else()
    set(CMAKE_C_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
    set(TVM_VISIBILITY_FLAG "")
    if (HIDE_PRIVATE_SYMBOLS)
      message(STATUS "Hide private symbols...")
      set(TVM_VISIBILITY_FLAG "-fvisibility=hidden")
    endif(HIDE_PRIVATE_SYMBOLS)
  endif ()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
  endif()

  # ld option to warn if symbols are undefined (e.g. libtvm_runtime.so
  # using symbols only present in libtvm.so).  Not needed for MSVC,
  # since this is already the default there.
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR ${CMAKE_SYSTEM_NAME} MATCHES "iOS")
    set(TVM_NO_UNDEFINED_SYMBOLS "-Wl,-undefined,error")
  else()
    set(TVM_NO_UNDEFINED_SYMBOLS "-Wl,--no-undefined")
  endif()
  message(STATUS "Forbidding undefined symbols in shared library, using ${TVM_NO_UNDEFINED_SYMBOLS} on platform ${CMAKE_SYSTEM_NAME}")

  # Detect if we're compiling for Android.
  set(TEST_FOR_ANDROID_CXX
      "#ifndef __ANDROID__"
      "#error"
      "#endif"
      "int main() {}")
  set(TEST_FOR_ANDROID_DIR
      "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp")
  set(TEST_FOR_ANDROID_FILE "${TEST_FOR_ANDROID_DIR}/test_for_android.cc")
  string(REPLACE ";" "\n" TEST_FOR_ANDROID_CXX_TEXT "${TEST_FOR_ANDROID_CXX}")
  file(WRITE "${TEST_FOR_ANDROID_FILE}" "${TEST_FOR_ANDROID_CXX_TEXT}")
  try_compile(BUILD_FOR_ANDROID "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}"
              "${TEST_FOR_ANDROID_FILE}")
  file(REMOVE "${TEST_FOR_ANDROID_FILE}")
  if(BUILD_FOR_ANDROID)
    message(STATUS "Building for Android")
  endif()
endif(MSVC)

list(APPEND TVM_RUNTIME_LINKER_LIBS ${CMAKE_DL_LIBS})

# add source group
tvm_file_glob(GLOB_RECURSE GROUP_SOURCE "src/*.cc")
tvm_file_glob(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
# Core compiler sources, extended by the backend modules.
tvm_file_glob(GLOB_RECURSE COMPILER_SRCS
    src/ir/*.cc
    src/sym/*.cc
    src/tirx/*.cc
    src/support/*.cc
    # Shared printer and TVMScript IR builder. Per-dialect printer/builder pieces
    # live under src/<dialect>/script/. List shared files explicitly so a dialect
    # implementation cannot be silently included through a recursive glob.
    src/script/ir_builder/base.cc
    src/script/ir_builder/ir.cc
    src/script/ir_builder/frame.cc
    src/script/printer/config.cc
    src/script/printer/doc.cc
    src/script/printer/doc_translator.cc
    src/script/printer/ir/call.cc
    src/script/printer/ir/utils.cc
    src/script/printer/ir/ir.cc
    src/script/printer/ir/misc.cc
    src/script/printer/ir/prim_type.cc
    src/script/printer/ir/prim_expr.cc
    src/script/printer/ir/stmt.cc
    src/script/printer/printer.cc
    src/script/printer/doc_printer.cc
    src/script/printer/script_printer.cc
    )

tvm_file_glob(GLOB CODEGEN_SRCS
  src/target/*.cc
  src/codegen/c/*.cc
  src/codegen/*.cc
  src/codegen/canonicalizer/*.cc
  src/codegen/llvm/target_kind.cc
  src/codegen/llvm/canonicalizer/*.cc
  src/backend/cuda/codegen/*.cc
  src/backend/cuda/op/*.cc
  src/backend/cuda/transforms/*.cc
  src/backend/cuda/tile/*.cc
  src/backend/cuda/layout/*.cc
  src/backend/hexagon/codegen/*.cc
  src/backend/metal/codegen/*.cc
  src/backend/metal/op/*.cc
  src/backend/opencl/codegen/*.cc
  src/backend/opencl/op/*.cc
  src/backend/rocm/codegen/*.cc
  src/backend/trn/codegen/*.cc
  src/backend/trn/op/*.cc
  src/backend/trn/transform/*.cc
  src/backend/trn/layout/*.cc
  src/backend/vulkan/codegen/target_kind.cc
  src/backend/vulkan/codegen/vulkan_fallback_module.cc
  src/backend/webgpu/codegen/*.cc
    )

list(APPEND COMPILER_SRCS ${CODEGEN_SRCS})
# Core runtime sources, extended by the backend modules.
tvm_file_glob(GLOB RUNTIME_SRCS
  src/runtime/*.cc
)
set(TVM_RUNTIME_EXT_OBJS "")

# Package runtime rules
if(NOT USE_RTTI)
endif()

if(INDEX_DEFAULT_I64)
  add_definitions(-DTVM_INDEX_DEFAULT_I64=1)
endif()

if(USE_KALLOC_ALIGNMENT)
  message(STATUS "Build Alloc alignment set to ${USE_KALLOC_ALIGNMENT}")
  add_definitions(-DTVM_KALLOC_ALIGNMENT=${USE_KALLOC_ALIGNMENT})
endif(USE_KALLOC_ALIGNMENT)

# Caches the build.
# Note that ccache-3.x doesn't support nvcc well, so CUDA kernels may never hit the cache and still
# need to be re-compiled every time. Using ccache 4.0+ can resolve this issue.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/CCache.cmake")

include(CheckCXXCompilerFlag)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

# Module rules
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/CUDA.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/OpenCL.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/OpenMP.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/Vulkan.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/Metal.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/ROCM.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/LLVM.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/contrib/Z3.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/Git.cmake")

include(GNUInstallDirs)

if(USE_THREADS)
  message(STATUS "Build with thread support...")
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
endif()

set(TVM_IS_DEBUG_BUILD OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR CMAKE_CXX_FLAGS MATCHES "-g")
  set(TVM_IS_DEBUG_BUILD ON)
  set(FILE_PREFIX_MAP_FLAG "-ffile-prefix-map=..=${CMAKE_CURRENT_SOURCE_DIR}")
  CHECK_CXX_COMPILER_FLAG("${FILE_PREFIX_MAP_FLAG}" FILE_PREFIX_MAP_SUPPORTED)
endif()

# Core runtime objects are shared by the dynamic and static libraries.
add_library(tvm_runtime_objs OBJECT ${RUNTIME_SRCS})
target_link_libraries(tvm_runtime_objs PUBLIC tvm_ffi_header)
if(TVM_VISIBILITY_FLAG)
  target_compile_options(tvm_runtime_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
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
target_link_libraries(tvm_runtime PUBLIC tvm_ffi::shared)
set_property(TARGET tvm_runtime APPEND PROPERTY LINK_OPTIONS "${TVM_NO_UNDEFINED_SYMBOLS}")
set_property(TARGET tvm_runtime APPEND PROPERTY LINK_OPTIONS "${TVM_VISIBILITY_FLAG}")

# Static consumers must retain registration objects from both archives, with
# tvm_ffi::static before tvm_runtime_static so FFI type attributes initialize first:
#   -Wl,--whole-archive tvm_ffi::static tvm_runtime_static -Wl,--no-whole-archive
# Use the corresponding archive-retention flags on other platforms.
set(_ffi_static_dependency tvm_ffi::shared)
if(TARGET tvm_ffi::static)
  set(_ffi_static_dependency tvm_ffi::static)
endif()
add_library(tvm_runtime_static STATIC
  $<TARGET_OBJECTS:tvm_runtime_objs>
  ${TVM_RUNTIME_EXT_OBJS}
)
target_link_libraries(tvm_runtime_static PUBLIC ${_ffi_static_dependency})
if(MSVC)
  # Static consumers reference ordinary symbols rather than DLL import stubs.
  target_compile_definitions(tvm_runtime_static INTERFACE
    TVM_RUNTIME_DLL= TVM_RUNTIME_DLL_EXPORT=)
  if(TARGET tvm_ffi::static)
    target_compile_definitions(tvm_runtime_static INTERFACE TVM_FFI_DLL= TVM_FFI_DLL_EXPORT=)
  endif()
endif()

set(_runtime_targets tvm_runtime tvm_runtime_static)

foreach(runtime_target IN LISTS _runtime_targets)
  target_include_directories(${runtime_target} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_link_libraries(${runtime_target} PRIVATE ${TVM_RUNTIME_LINKER_LIBS})
  if(USE_THREADS)
    target_link_libraries(${runtime_target} PUBLIC Threads::Threads)
  endif()
endforeach()

set_target_properties(${_runtime_targets} PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
if(MSVC)
  foreach(config_type Release RELEASE Debug DEBUG RelWithDebInfo RELWITHDEBINFO MinSizeRel MINSIZEREL)
    set_target_properties(${_runtime_targets} PROPERTIES
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
if(NOT IS_SUBPROJECT)
  add_custom_target(runtime DEPENDS ${_runtime_targets})
endif()
install(TARGETS tvm_runtime DESTINATION lib${LIB_SUFFIX})
# Keep wheel contents unchanged; the archive is available to native consumers.
if(TARGET tvm_runtime_static AND NOT TVM_BUILD_PYTHON_MODULE)
  install(TARGETS tvm_runtime_static DESTINATION lib${LIB_SUFFIX})
endif()
add_library(tvm_tirx_objs OBJECT ${COMPILER_SRCS})
target_link_libraries(tvm_tirx_objs PUBLIC tvm_ffi_header)
if(TARGET tvm_llvm_header)
  target_link_libraries(tvm_tirx_objs PUBLIC tvm_llvm_header)
endif()
if(TVM_VISIBILITY_FLAG)
  target_compile_options(tvm_tirx_objs PRIVATE "${TVM_VISIBILITY_FLAG}")
endif()
if(MSVC)
  # Compiler objects import runtime symbols from the separate runtime DLL.
  target_compile_definitions(tvm_tirx_objs PRIVATE TVM_TIRX_EXPORTS TVM_FFI_EXPORTS)
endif()

# Compiler library: built from compiler-only sources.
# Links against tvm_runtime (and tvm_ffi::shared transitively). Loaded with
# RTLD_LOCAL so compiler-internal symbols don't leak into the global namespace.
add_library(tvm_tirx SHARED
  $<TARGET_OBJECTS:tvm_tirx_objs>
)
target_link_libraries(tvm_tirx PUBLIC tvm_runtime tvm_ffi::shared)
target_include_directories(tvm_tirx PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>" "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
set_property(TARGET tvm_tirx APPEND PROPERTY LINK_OPTIONS "${TVM_NO_UNDEFINED_SYMBOLS}")
set_property(TARGET tvm_tirx APPEND PROPERTY LINK_OPTIONS "${TVM_VISIBILITY_FLAG}")

# Work around a GNU ld (binutils) relaxation bug that miscompiles
# R_X86_64_GOTPCRELX relocations inside very large statically-linked archives.
# When the full LLVM static libraries are linked into libtvm_tirx.so, the
# library is large enough that ld can relax an indirect GOT call (LLVM built
# with -fno-plt emits these) into a direct call with an incorrect displacement.
# The call then targets read-only data instead of the intended function and
# crashes at runtime with a SIGSEGV inside llvm::X86Subtarget during code
# generation. Disabling linker relaxation keeps the GOT-indirect sequences and
# avoids the miscompilation; it is harmless when LLVM is linked dynamically.
# See binutils bug ld/25754.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT "${USE_LLVM}" MATCHES ${IS_FALSE_PATTERN})
  set_property(TARGET tvm_tirx APPEND PROPERTY LINK_OPTIONS "-Wl,--no-relax")
  # LLVM's --system-libs reports -lxml2, but TVM calls no libxml2 symbols. The
  # manylinux toolchain does not default to --as-needed, so set it explicitly:
  # the unused libxml2 is then not recorded as a NEEDED dependency, so auditwheel
  # does not vendor it (and its deps) into the wheel.
  set_property(TARGET tvm_tirx APPEND PROPERTY LINK_OPTIONS "-Wl,--as-needed")
endif()

set_target_properties(tvm_tirx PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
if(MSVC)
  foreach(config_type Release RELEASE Debug DEBUG RelWithDebInfo RELWITHDEBINFO MinSizeRel MINSIZEREL)
    set_target_properties(tvm_tirx PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
    )
  endforeach()
endif()

if(USE_THREADS)
  target_link_libraries(tvm_tirx PUBLIC Threads::Threads)
endif()
target_link_libraries(tvm_tirx PRIVATE ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})

if(TVM_IS_DEBUG_BUILD)
  target_compile_options(tvm_tirx PRIVATE "${FILE_PREFIX_MAP_FLAG}")
  if(FILE_PREFIX_MAP_SUPPORTED)
    target_compile_options(tvm_tirx PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
    target_compile_options(tvm_tirx_objs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FILE_PREFIX_MAP_FLAG}>)
  endif()
endif()

tvm_ffi_add_apple_dsymutil(tvm_tirx)
tvm_configure_target_library(tvm_tirx)
install(TARGETS tvm_tirx DESTINATION lib${LIB_SUFFIX})

# logging option for libbacktrace
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/Logging.cmake")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/ClangFlags.cmake")
if(USE_CUDA AND USE_NVTX)
  add_compile_definitions(TVM_NVTX_ENABLED=1)
  target_compile_definitions(tvm_runtime INTERFACE TVM_NVTX_ENABLED=1)
endif()
if(INDEX_DEFAULT_I64)
  target_compile_definitions(tvm_runtime INTERFACE TVM_INDEX_DEFAULT_I64=1)
endif()
if(USE_KALLOC_ALIGNMENT)
  target_compile_definitions(tvm_runtime INTERFACE TVM_KALLOC_ALIGNMENT=${USE_KALLOC_ALIGNMENT})
endif()
if(TVM_DEBUG_WITH_ABI_CHANGE)
  target_compile_definitions(tvm_runtime PUBLIC TVM_DEBUG_WITH_ABI_CHANGE)
  target_compile_definitions(tvm_runtime_objs PRIVATE TVM_DEBUG_WITH_ABI_CHANGE)
  target_compile_definitions(tvm_tirx_objs PRIVATE TVM_DEBUG_WITH_ABI_CHANGE)
endif()

get_target_property(_runtime_compile_definitions tvm_runtime INTERFACE_COMPILE_DEFINITIONS)
if(_runtime_compile_definitions)
  target_compile_definitions(tvm_runtime_static INTERFACE ${_runtime_compile_definitions})
endif()
add_library(tvm_tirx_private_headers INTERFACE)
set_target_properties(tvm_tirx_private_headers PROPERTIES EXPORT_NAME private_headers)
target_include_directories(tvm_tirx_private_headers INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
  "$<BUILD_INTERFACE:${COMPILER_RT_PATH}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tvm_tirx/private>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tvm_tirx/compiler-rt>")
set(_tirx_targets tvm_tirx ${_runtime_targets} tvm_tirx_private_headers ${TVM_RUNTIME_BACKEND_LIBS})
foreach(_target IN LISTS _tirx_targets)
  get_target_property(_export_name ${_target} EXPORT_NAME)
  if(NOT _export_name)
    set(_export_name ${_target})
  endif()
  add_library(tvm_tirx::${_export_name} ALIAS ${_target})
endforeach()
install(TARGETS ${_tirx_targets} EXPORT tvm_tirxTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
# CLML's retained OpenCL interface needs this bounded private include closure.
foreach(_header backend/opencl/memory_scope.h
    backend/opencl/runtime/opencl_common.h backend/opencl/runtime/texture.h
    runtime/metadata.h runtime/pack_args.h runtime/workspace_pool.h)
  get_filename_component(_dir "${_header}" DIRECTORY)
  install(FILES "src/${_header}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tvm_tirx/private/${_dir}")
endforeach()
install(FILES 3rdparty/compiler-rt/builtin_fp16.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm_tirx/compiler-rt)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/tvm_tirxConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/tvm_tirxConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx")
install(EXPORT tvm_tirxTargets NAMESPACE tvm_tirx:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tvm_tirxConfig.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx")
install(DIRECTORY cmake/utils cmake/modules DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx" FILES_MATCHING PATTERN "*.cmake")
set(TVM_TIRX_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(IS_SUBPROJECT)
  set(TVM_TIRX_CMAKE_DIR "${TVM_TIRX_CMAKE_DIR}" PARENT_SCOPE)
  foreach(_option USE_CUDA USE_ROCM BUILD_FOR_ANDROID)
    set(TVM_TIRX_${_option} "${${_option}}" PARENT_SCOPE)
  endforeach()
endif()
add_subdirectory(tests/cpp)
if(GTEST_FOUND)
  enable_testing()
  include(CTest)
endif()
find_and_set_linker(${USE_ALTERNATIVE_LINKER})
if(SUMMARIZE)
  print_summary()
endif()
dump_options_to_file("${TVM_ALL_OPTIONS}")
if(TVM_BUILD_PYTHON_MODULE AND NOT IS_SUBPROJECT)
  foreach(_extra_lib IN LISTS TVM_PACKAGE_EXTRA_LIBS)
    if(NOT EXISTS "${_extra_lib}")
      message(FATAL_ERROR "TVM_PACKAGE_EXTRA_LIBS entry does not exist: ${_extra_lib}")
    endif()
    install(FILES "${_extra_lib}" DESTINATION lib)
  endforeach()
  install(DIRECTORY licenses/ DESTINATION licenses)
  install(FILES LICENSE NOTICE DESTINATION .)
endif()
