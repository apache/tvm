cmake_minimum_required(VERSION 3.14)

project(
  tvm_ffi
  VERSION 1.0
  DESCRIPTION "TVM's FFI system"
  LANGUAGES CXX
)

option(TVM_FFI_ALLOW_DYN_TYPE "Support for objects with non-static type indices. When turned on, targets linked against `tvm_ffi` will allow objects that comes with non-pre-defined type indices, so that the object hierarchy could expand without limitation. This will require the downstream targets to link against target `tvm_ffi_registry` to be effective." OFF)
option(TVM_FFI_BUILD_TESTS "Adding test targets." OFF)

include(cmake/Utils/CxxWarning.cmake)
include(cmake/Utils/Sanitizer.cmake)

########## Target: `dlpack_header` ##########

add_library(dlpack_header INTERFACE)
target_include_directories(dlpack_header INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include")

########## Target: `tvm_ffi_registry_{objs|static|shared}` ##########

add_library(
  tvm_ffi_registry_objs
  EXCLUDE_FROM_ALL
  OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/registry.cc"
)
set_target_properties(
  tvm_ffi_registry_objs PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
add_cxx_warning(tvm_ffi_registry_objs)
target_link_libraries(tvm_ffi_registry_objs PRIVATE dlpack_header)
target_include_directories(tvm_ffi_registry_objs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(tvm_ffi_registry_objs PRIVATE TVM_FFI_ALLOW_DYN_TYPE=1)
if (MSVC)
  target_compile_definitions(tvm_ffi_registry_objs PRIVATE TVM_FFI_EXPORTS)
endif()

add_library(tvm_ffi_registry_static EXCLUDE_FROM_ALL STATIC $<TARGET_OBJECTS:tvm_ffi_registry_objs>)
add_library(tvm_ffi_registry_shared EXCLUDE_FROM_ALL SHARED $<TARGET_OBJECTS:tvm_ffi_registry_objs>)
set_target_properties(tvm_ffi_registry_shared tvm_ffi_registry_static tvm_ffi_registry_objs
    PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

########## Target: `tvm_ffi` ##########

add_library(tvm_ffi INTERFACE)
target_link_libraries(tvm_ffi INTERFACE dlpack_header)
target_compile_features(tvm_ffi INTERFACE cxx_std_17)
target_include_directories(tvm_ffi INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if (TVM_FFI_ALLOW_DYN_TYPE)
  message(STATUS "Setting C++ macro TVM_FFI_ALLOW_DYN_TYPE - 1")
  target_compile_definitions(tvm_ffi INTERFACE TVM_FFI_ALLOW_DYN_TYPE=1)
else()
  message(STATUS "Setting C++ macro TVM_FFI_ALLOW_DYN_TYPE - 0")
  target_compile_definitions(tvm_ffi INTERFACE TVM_FFI_ALLOW_DYN_TYPE=0)
endif()

########## Adding tests ##########

if (${PROJECT_NAME} STREQUAL ${CMAKE_PROJECT_NAME})
  if (TVM_FFI_BUILD_TESTS)
    enable_testing()
    message(STATUS "Enable Testing")
    include(cmake/Utils/AddGoogleTest.cmake)
    add_subdirectory(tests/example/)
  endif()
endif ()
