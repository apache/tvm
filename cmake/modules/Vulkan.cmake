# Be compatible with older version of CMake
if(NOT $ENV{VULKAN_SDK} STREQUAL "")
  set(Vulkan_INCLUDE_DIRS $ENV{VULKAN_SDK}/include)
  set(Vulkan_FOUND ON)
else()
  find_package(Vulkan QUIET)
endif()

if(Vulkan_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
  include_directories(${Vulkan_INCLUDE_DIRS})
endif(Vulkan_FOUND)

if(USE_VULKAN)
  if(NOT $ENV{VULKAN_SDK} STREQUAL "")
    find_library(Vulkan_LIBRARY vulkan $ENV{VULKAN_SDK}/lib)
  else()
    find_package(Vulkan REQUIRED)
  endif()
  message(STATUS "Build with VULKAN support")
  file(GLOB RUNTIME_VULKAN_SRCS src/runtime/vulkan/*.cc)
  file(GLOB COMPILER_VULKAN_SRCS src/codegen/spirv/*.cc)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${Vulkan_LIBRARY})
  list(APPEND RUNTIME_SRCS ${RUNTIME_VULKAN_SRCS})
  list(APPEND COMPILER_SRCS ${COMPILER_VULKAN_SRCS})
  get_filename_component(VULKAN_LIB_PATH ${Vulkan_LIBRARY} DIRECTORY)
  find_library(SPIRV_TOOLS_LIB SPIRV-Tools
               ${VULKAN_LIB_PATH}/spirv-tools)
  list(APPEND TVM_LINKER_LIBS ${SPIRV_TOOLS_LIB})
endif(USE_VULKAN)
