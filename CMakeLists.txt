cmake_minimum_required(VERSION 3.7)
project(tvm C CXX)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/build/private/local_config.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/build/private/local_config.cmake)
endif()

include(cmake/Util.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.

tvm_option(USE_CUDA "Build with CUDA" OFF)
tvm_option(USE_OPENCL "Build with OpenCL" OFF)
tvm_option(USE_VULKAN "Build with Vulkan" OFF)
tvm_option(USE_OPENGL "Build with OpenGL" OFF)
tvm_option(USE_METAL "Build with Metal" OFF)
tvm_option(USE_RPC "Build with RPC" ON)
tvm_option(USE_GRAPH_RUNTIME "Build with tiny graph runtime" ON)
tvm_option(USE_LLVM "Build with LLVM" OFF)
tvm_option(USE_RTTI "Build with RTTI" ON)
tvm_option(USE_MSVC_MT "Build with MT" OFF)
tvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)

tvm_option(USE_CUDNN "Build with cuDNN" OFF)

include_directories("include")
include_directories("HalideIR/src")
include_directories("dlpack/include")


set(TVM_LINKER_LIBS "")
set(TVM_RUNTIME_LINKER_LIBS "")

# compile
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-DHalide_SHARED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
  set(CMAKE_C_FLAGS "-O3 -Wall -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
endif(MSVC)

# add source group
FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc" "HalideIR/src/*.cpp")
FILE(GLOB_RECURSE GROUP_Include "src/*.h" "include/*.h" "HalideIR/src/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_Include})

file(GLOB COMPILER_SRCS
    src/api/*.cc
    src/arithmetic/*.cc
    src/codegen/*.cc
    src/codegen/stack_vm/*.cc
    src/lang/*.cc
    src/pass/*.cc
    src/op/*.cc
    src/schedule/*.cc
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/topi/include)
file(GLOB TOPI_SRCS
    topi/src/*.cc
)
file(GLOB_RECURSE HALIDEIR_SRCS HalideIR/src/*.cpp)
list(APPEND COMPILER_SRCS ${HALIDEIR_SRCS})
file(GLOB RUNTIME_SRCS src/runtime/*.cc)
file(GLOB COMPILER_LLVM_SRCS src/codegen/llvm/*.cc)
file(GLOB COMPILER_VULKAN_SRCS src/codegen/spirv/*.cc)
file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
file(GLOB RUNTIME_OPENCL_SRCS src/runtime/opencl/*.cc)
file(GLOB RUNTIME_OPENGL_SRCS src/runtime/opengl/*.cc)
file(GLOB RUNTIME_VULKAN_SRCS src/runtime/vulkan/*.cc)
file(GLOB RUNTIME_METAL_SRCS src/runtime/metal/*.mm)
file(GLOB RUNTIME_RPC_SRCS src/runtime/rpc/*.cc)
file(GLOB RUNTIME_GRAPH_SRCS src/runtime/graph/*.cc)

if(USE_CUDA)
find_package(CUDA)
# Find CUDA doesn't find all the libraries we need, add the extra ones
find_library(CUDA_CUDA_LIBRARIES cuda
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
find_library(CUDA_NVRTC_LIBRARIES nvrtc
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
  set(CUDA_CUDA_LIBRARY ${CUDA_CUDA_LIBRARIES})

  find_package(CUDA QUIET REQUIRED)
  message(STATUS "Build with CUDA support")
	include_directories(${CUDA_INCLUDE_DIRS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})
  if(MSVC)
    find_library(CUDA_NVRTC_LIB nvrtc
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
    list(APPEND TVM_LINKER_LIBS ${CUDA_NVRTC_LIB})
  else(MSVC)
    find_library(CUDA_NVRTC_LIB nvrtc
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64
      ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    list(APPEND TVM_LINKER_LIBS ${CUDA_NVRTC_LIB})
  endif(MSVC)

  if(USE_CUDNN)
    message(STATUS "Build with cuDNN support")
    file(GLOB CONTRIB_CUDNN_SRCS src/contrib/cudnn/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUDNN_SRCS})
    if(MSVC)
      find_library(CUDA_CUDNN_LIB cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
      list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIB})
    else(MSVC)
      find_library(CUDA_CUDNN_LIB cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)
      list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIB})
    endif(MSVC)
  endif(USE_CUDNN)

  add_definitions(-DTVM_CUDA_RUNTIME=1)
else(USE_CUDA)
  add_definitions(-DTVM_CUDA_RUNTIME=0)
endif(USE_CUDA)

if(USE_OPENCL)
  find_package(OpenCL QUIET REQUIRED)
  message(STATUS "Build with OpenCL support")
  include_directories(${OpenCL_INCLUDE_DIRS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${OpenCL_LIBRARIES})
  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENCL_SRCS})
  add_definitions(-DTVM_OPENCL_RUNTIME=1)
else(USE_OPENCL)
  add_definitions(-DTVM_OPENCL_RUNTIME=0)
endif(USE_OPENCL)

if(USE_OPENGL)
  find_package(OpenGL QUIET REQUIRED)
  find_package(glfw3 QUIET REQUIRED)
  message(STATUS "Build with OpenGL support")
  include_directories(${OPENGL_INCLUDE_DIRS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${OpenGL_LIBRARIES} glfw)
  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENGL_SRCS})
  add_definitions(-DTVM_OPENGL_RUNTIME=1)
else(USE_OPENGL)
  add_definitions(-DTVM_OPENGL_RUNTIME=0)
endif(USE_OPENGL)

if(USE_VULKAN)
  find_package(Vulkan REQUIRED)
  message(STATUS "Build with VULKAN support")
  include_directories(${Vulkan_INCLUDE_DIRS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${Vulkan_LIBRARIES})
  list(APPEND RUNTIME_SRCS ${RUNTIME_VULKAN_SRCS})
  list(APPEND COMPILER_SRCS ${COMPILER_VULKAN_SRCS})
  get_filename_component(VULKAN_LIB_PATH ${Vulkan_LIBRARY} DIRECTORY)
  find_library(SPIRV_TOOLS_LIB SPIRV-Tools
               ${VULKAN_LIB_PATH}/spirv-tools)
  list(APPEND TVM_LINKER_LIBS ${SPIRV_TOOLS_LIB})
  add_definitions(-DTVM_VULKAN_RUNTIME=1)
else(USE_VULKAN)
  add_definitions(-DTVM_VULKAN_RUNTIME=0)
endif(USE_VULKAN)

if(USE_METAL)
  find_package(OpenCL QUIET REQUIRED)
  message(STATUS "Build with Metal support")
  FIND_LIBRARY(METAL_LIB Metal)
  FIND_LIBRARY(FOUNDATION_LIB Foundation)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${METAL_LIB} ${FOUNDATION_LIB})
  list(APPEND RUNTIME_SRCS ${RUNTIME_METAL_SRCS})
  add_definitions(-DTVM_METAL_RUNTIME=1)
else(USE_METAL)
  add_definitions(-DTVM_METAL_RUNTIME=0)
endif(USE_METAL)

if(USE_RPC)
  message(STATUS "Build with RPC support...")
  list(APPEND RUNTIME_SRCS ${RUNTIME_RPC_SRCS})
endif(USE_RPC)

if(USE_GRAPH_RUNTIME)
  message(STATUS "Build with Graph runtime support...")
  list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_SRCS})
endif(USE_GRAPH_RUNTIME)

if(USE_LLVM)
  find_package(LLVM CONFIG REQUIRED)
  include_directories(${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
  set(TVM_LLVM_VERSION ${LLVM_VERSION_MAJOR}${LLVM_VERSION_MINOR})
  message(STATUS "Build with LLVM " ${LLVM_PACKAGE_VERSION})
  message(STATUS "Set TVM_LLVM_VERSION=" ${TVM_LLVM_VERSION})
  add_definitions(-DTVM_LLVM_VERSION=${TVM_LLVM_VERSION})
  add_definitions(-DDMLC_USE_FOPEN64=0)
  llvm_map_components_to_libnames(LLVM_LIBS all)
  list(REMOVE_ITEM LLVM_LIBS LTO)
  list(APPEND TVM_LINKER_LIBS ${LLVM_LIBS})
  list(APPEND COMPILER_SRCS ${COMPILER_LLVM_SRCS})
  if(NOT MSVC)
    set_property(SOURCE ${COMPILER_LLVM_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS
      "-fno-rtti -DDMLC_ENABLE_RTTI=0")
  endif()
endif(USE_LLVM)

if(NOT USE_RTTI)
  add_definitions(-DDMLC_ENABLE_RTTI=0)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dmlc-core/CMakeLists.txt)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dmlc-core/include)
  if (INSTALL_DEV)
    install(
      DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dmlc-core/include/." DESTINATION "include"
      FILES_MATCHING
      PATTERN "*.h"
    )
  endif()
elseif(DMLC_CORE_PATH)
  include_directories(${DMLC_CORE_PATH}/include)
endif()

list(APPEND RUNTIME_SRCS ${GROUP_Include})
add_library(tvm SHARED ${COMPILER_SRCS} ${RUNTIME_SRCS})
add_library(tvm_topi SHARED ${TOPI_SRCS})
add_library(tvm_runtime SHARED ${RUNTIME_SRCS})
target_link_libraries(tvm ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(tvm_topi tvm ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(tvm_runtime  ${TVM_RUNTIME_LINKER_LIBS})
install(TARGETS tvm_runtime DESTINATION lib${LIB_SUFFIX})
if (INSTALL_DEV)
  install(TARGETS tvm DESTINATION lib${LIB_SUFFIX})
  install(
    DIRECTORY "include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
  )
  install(
    DIRECTORY "topi/include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
  )
  install(
    DIRECTORY "HalideIR/src/." DESTINATION "include/HalideIR"
    FILES_MATCHING
    PATTERN "*.h"
  )
  install(
    DIRECTORY "dlpack/include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
  )
else(INSTALL_DEV)
  install(
    DIRECTORY "include/tvm/runtime/." DESTINATION "include/tvm/runtime"
    FILES_MATCHING
    PATTERN "*.h"
  )
endif(INSTALL_DEV)

if(MSVC)
  target_compile_definitions(tvm PRIVATE -DHalide_EXPORTS)
  target_compile_definitions(tvm_runtime PRIVATE -DHalide_EXPORTS)
  target_compile_definitions(tvm PRIVATE -DTVM_EXPORTS)
  target_compile_definitions(tvm_runtime PRIVATE -DTVM_EXPORTS)
endif()
