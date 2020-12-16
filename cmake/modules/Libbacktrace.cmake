include(ExternalProject)

# Git tag for libbacktrace
set(TAG dedbe13fda00253fe5d4f2fb812c909729ed5937)

ExternalProject_Add(project_libbacktrace
  PREFIX libbacktrace
  GIT_REPOSITORY https://github.com/ianlancetaylor/libbacktrace.git
  GIT_TAG ${TAG}
  CONFIGURE_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/src/project_libbacktrace/configure"
                    "--prefix=${CMAKE_CURRENT_BINARY_DIR}/libbacktrace"
  INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace"
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/lib/libbacktrace.a"
                   "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include/backtrace.h"
  # disable the builtin update because it rebuilds on every build
  UPDATE_DISCONNECTED ON
  UPDATE_COMMAND ""
  # libbacktrace has a bug on macOS with shared libraries.
  PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/libbacktrace_macos.patch
  )

# Only rebuild libbacktrace if this file changes
ExternalProject_Add_Step(project_libbacktrace update-new
  DEPENDERS configure
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/Libbacktrace.cmake"
  COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/src/project_libbacktrace; git checkout -f ${TAG}
  )

add_library(libbacktrace STATIC IMPORTED)
add_dependencies(libbacktrace project_libbacktrace)
set_property(TARGET libbacktrace
  PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/lib/libbacktrace.a)
# create include directory so cmake doesn't complain
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include)
target_include_directories(libbacktrace
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include>)
