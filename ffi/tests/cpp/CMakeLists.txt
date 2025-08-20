file(GLOB _test_sources "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
file(GLOB _test_extra_sources "${CMAKE_CURRENT_SOURCE_DIR}/extra/test*.cc")

if (TVM_FFI_USE_EXTRA_CXX_API)
  list(APPEND _test_sources ${_test_extra_sources})
endif()

add_executable(
  tvm_ffi_tests
  EXCLUDE_FROM_ALL
  ${_test_sources}
)
set_target_properties(
  tvm_ffi_tests PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
tvm_ffi_add_cxx_warning(tvm_ffi_tests)
add_sanitizer_address(tvm_ffi_tests)
tvm_ffi_add_apple_dsymutil(tvm_ffi_tests)
tvm_ffi_add_msvc_flags(tvm_ffi_tests)
target_link_libraries(tvm_ffi_tests PRIVATE tvm_ffi_shared)
tvm_ffi_add_googletest(tvm_ffi_tests)

if (MSVC)
  target_link_options(tvm_ffi_tests PRIVATE /DEBUG)
endif()
