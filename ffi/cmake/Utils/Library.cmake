function(add_target_from_obj target_name obj_target_name)
  add_library(${target_name}_static STATIC $<TARGET_OBJECTS:${obj_target_name}>)
  set_target_properties(
    ${target_name}_static PROPERTIES
    OUTPUT_NAME "${target_name}_static"
    PREFIX "lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
  add_library(${target_name}_shared SHARED $<TARGET_OBJECTS:${obj_target_name}>)
  set_target_properties(
    ${target_name}_shared PROPERTIES
    OUTPUT_NAME "${target_name}"
    PREFIX "lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  add_custom_target(${target_name})
  add_dependencies(${target_name} ${target_name}_static ${target_name}_shared)
  if (MSVC)
    target_compile_definitions(${obj_target_name} PRIVATE TVM_FFI_EXPORTS)
    set_target_properties(
      ${obj_target_name} ${target_name}_shared ${target_name}_static
      PROPERTIES
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
  endif()
endfunction()
