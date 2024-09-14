file(GLOB _test_sources "${CMAKE_CURRENT_SOURCE_DIR}/test*.cc")
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
add_cxx_warning(tvm_ffi_tests)
add_sanitizer_address(tvm_ffi_tests)
target_link_libraries(tvm_ffi_tests PRIVATE tvm_ffi)

if (TVM_FFI_BUILD_REGISTRY)
  target_link_libraries(tvm_ffi_tests PRIVATE tvm_ffi_registry_shared)
endif()

add_googletest(tvm_ffi_tests)
