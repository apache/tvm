function(add_cxx_warning target_name)
  # GNU, Clang, or AppleClang
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(${target_name} PRIVATE "-Werror" "-Wall" "-Wextra" "-Wpedantic")
    return()
  endif()
  # MSVC
  if(MSVC)
    target_compile_options(${target_name} PRIVATE "/W4" "/WX")
    return()
  endif()
  message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endfunction()
