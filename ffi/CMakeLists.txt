cmake_minimum_required(VERSION 3.14)

project(
  tvm_ffi
  VERSION 1.0
  DESCRIPTION "TVM's FFI system"
  LANGUAGES CXX C
)

option(TVM_FFI_BUILD_TESTS "Adding test targets." OFF)

########## NOTE: all options below are related to dynamic registry #####
option(TVM_FFI_BUILD_REGISTRY
  "Support for objects with non-static type indices. When turned on, \
  targets linked against `tvm_ffi` will allow objects that comes with non-pre-defined type indices, \
  as well as getting full stacktrace during debugging. \
  so that the object hierarchy could expand without limitation. \
  This will require the downstream targets to link against target `tvm_ffi` to be effective."
  OFF
)
option(TVM_FFI_USE_LIBBRACKTRACE "Enable libbacktrace" ON)
option(TVM_FFI_BACKTRACE_ON_SEGFAULT "Set signal handler to print traceback on segfault" ON)

include(cmake/Utils/CxxWarning.cmake)
include(cmake/Utils/Sanitizer.cmake)
include(cmake/Utils/Library.cmake)
include(cmake/Utils/AddLibbacktrace.cmake)

########## Target: `dlpack_header` ##########

add_library(dlpack_header INTERFACE)
target_include_directories(dlpack_header INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include")

########## Target: `tvm_ffi_header` ##########

add_library(tvm_ffi_header INTERFACE)
target_include_directories(tvm_ffi_header INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(tvm_ffi_header INTERFACE dlpack_header)

########## Target: `tvm_ffi` ##########
add_library(tvm_ffi_objs OBJECT
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/traceback.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/object.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/function.cc"
)
set_target_properties(
  tvm_ffi_objs PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  PREFIX "lib"
)
add_cxx_warning(tvm_ffi_objs)
target_link_libraries(tvm_ffi_objs PRIVATE dlpack_header)
target_include_directories(tvm_ffi_objs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if (TVM_FFI_USE_LIBBRACKTRACE)
  message(STATUS "Setting C++ macro TVM_FFI_USE_LIBBRACKTRACE - 1")
  target_compile_definitions(tvm_ffi_objs PRIVATE TVM_FFI_USE_LIBBRACKTRACE=1)
else()
  message(STATUS "Setting C++ macro TVM_FFI_USE_LIBBRACKTRACE - 0")
  target_compile_definitions(tvm_ffi_objs PRIVATE TVM_FFI_USE_LIBBRACKTRACE=0)
endif()

if (TVM_FFI_BACKTRACE_ON_SEGFAULT)
  message(STATUS "Setting C++ macro TVM_FFI_BACKTRACE_ON_SEGFAULT - 1")
  target_compile_definitions(tvm_ffi_objs PRIVATE TVM_FFI_BACKTRACE_ON_SEGFAULT=1)
else()
  message(STATUS "Setting C++ macro TVM_FFI_BACKTRACE_ON_SEGFAULT - 0")
  target_compile_definitions(tvm_ffi_objs PRIVATE TVM_FFI_BACKTRACE_ON_SEGFAULT=0)
endif()

add_target_from_obj(tvm_ffi tvm_ffi_objs)

if (TARGET libbacktrace)
  target_link_libraries(tvm_ffi_objs PRIVATE libbacktrace)
  target_link_libraries(tvm_ffi_shared PRIVATE libbacktrace)
  target_link_libraries(tvm_ffi_static PRIVATE libbacktrace)
endif ()
target_link_libraries(tvm_ffi_objs PUBLIC tvm_ffi_header)
target_link_libraries(tvm_ffi_shared PUBLIC tvm_ffi_header)
target_link_libraries(tvm_ffi_static PUBLIC tvm_ffi_header)

install(TARGETS tvm_ffi_static DESTINATION lib${LIB_SUFFIX})
install(TARGETS tvm_ffi_shared DESTINATION lib${LIB_SUFFIX})

########## Adding tests ##########

if (${PROJECT_NAME} STREQUAL ${CMAKE_PROJECT_NAME})
  if (TVM_FFI_BUILD_TESTS)
    enable_testing()
    message(STATUS "Enable Testing")
    include(cmake/Utils/AddGoogleTest.cmake)
    add_subdirectory(tests/cpp/)
  endif()
endif ()
