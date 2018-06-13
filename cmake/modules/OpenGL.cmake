find_package(OpenGL QUIET)

if(OpenGL_FOUND)
  # always set the includedir when dir is available
  # avoid global retrigger of cmake
  include_directories(${OPENGL_INCLUDE_DIRS})
endif(OpenGL_FOUND)

if(USE_OPENGL)
  find_package(OpenGL REQUIRED)
  find_package(glfw3 QUIET REQUIRED)
  message(STATUS "Build with OpenGL support")
  file(GLOB RUNTIME_OPENGL_SRCS src/runtime/opengl/*.cc)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${OpenGL_LIBRARIES} glfw)
  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENGL_SRCS})
else(USE_OPENGL)
  list(APPEND COMPILER_SRCS src/codegen/opt/build_opengl_off.cc)
endif(USE_OPENGL)
