# Be compatible with older version of CMake
find_vulkan(${USE_VULKAN})

if(Vulkan_FOUND)
  # always set the includedir
  # avoid global retrigger of cmake
  include_directories(${Vulkan_INCLUDE_DIRS})
endif(Vulkan_FOUND)

if(USE_VULKAN)
  if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Cannot find Vulkan, USE_VULKAN=" ${USE_VULKAN})
  endif()
  message(STATUS "Build with VULKAN support")
  file(GLOB RUNTIME_VULKAN_SRCS src/runtime/vulkan/*.cc)
  file(GLOB COMPILER_VULKAN_SRCS src/codegen/spirv/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_VULKAN_SRCS})
  list(APPEND COMPILER_SRCS ${COMPILER_VULKAN_SRCS})

  list(APPEND TVM_LINKER_LIBS ${Vulkan_SPIRV_TOOLS_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${Vulkan_LIBRARY})
endif(USE_VULKAN)
