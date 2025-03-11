include(ExternalProject)

function(_libbacktrace_compile)
  set(_libbacktrace_source ${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/libbacktrace)
  set(_libbacktrace_prefix ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace)
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin" AND (CMAKE_C_COMPILER MATCHES "^/Library" OR CMAKE_C_COMPILER MATCHES "^/Applications"))
    set(_cmake_c_compiler "/usr/bin/cc")
  else()
    set(_cmake_c_compiler "${CMAKE_C_COMPILER}")
  endif()

  message(STATUS CMAKC_C_COMPILER="${CMAKE_C_COMPILER}")

  file(MAKE_DIRECTORY ${_libbacktrace_prefix}/include)
  file(MAKE_DIRECTORY ${_libbacktrace_prefix}/lib)

  ExternalProject_Add(project_libbacktrace
    PREFIX libbacktrace
    SOURCE_DIR ${_libbacktrace_source}
    BINARY_DIR ${_libbacktrace_prefix}
    CONFIGURE_COMMAND
      "${_libbacktrace_source}/configure"
      "--prefix=${_libbacktrace_prefix}"
      --with-pic
      "CC=${_cmake_c_compiler}"
      "CPP=${_cmake_c_compiler} -E"
      "CFLAGS=${CMAKE_C_FLAGS}"
      "LDFLAGS=${CMAKE_EXE_LINKER_FLAGS}"
      "NM=${CMAKE_NM}"
      "STRIP=${CMAKE_STRIP}"
      "--host=${MACHINE_NAME}"
  INSTALL_DIR ${_libbacktrace_prefix}
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_BYPRODUCTS "${_libbacktrace_prefix}/lib/libbacktrace.a"
                   "${_libbacktrace_prefix}/include/backtrace.h"
  )
  ExternalProject_Add_Step(project_libbacktrace checkout DEPENDERS configure DEPENDEES download)
  set_target_properties(project_libbacktrace PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_library(libbacktrace STATIC IMPORTED)
  add_dependencies(libbacktrace project_libbacktrace)
  set_target_properties(libbacktrace PROPERTIES
    IMPORTED_LOCATION ${_libbacktrace_prefix}/lib/libbacktrace.a
    INTERFACE_INCLUDE_DIRECTORIES ${_libbacktrace_prefix}/include
  )
endfunction()

if(NOT MSVC)
  _libbacktrace_compile()
endif()
