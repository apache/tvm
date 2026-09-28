cmake_policy(SET CMP0069 NEW) # suppress cmake warning about IPO

set(RTVM_SOURCES
  main.cc
  tvm_runner.cc
)
set(TVM_RUNNER_SOURCES
  tvm_runner.cc
)

set(RTVM_LINKER_LIBS "")

# Set output to same directory as the other TVM libs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(rtvm ${RTVM_SOURCES})
add_library(tvm_runner_objs OBJECT ${TVM_RUNNER_SOURCES})
add_library(tvm_runner SHARED $<TARGET_OBJECTS:tvm_runner_objs>)

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET rtvm PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(WIN32)
  target_compile_definitions(rtvm PUBLIC -DNOMINMAX)
endif()

if (OS)
   if (OS STREQUAL "Linux")
      set_property(TARGET rtvm PROPERTY LINK_FLAGS -lpthread)
      set_property(TARGET tvm_runner PROPERTY LINK_FLAGS -lpthread)
   endif()
endif()

if(USE_OPENCL)
   if (ANDROID_ABI)
     if(DEFINED ENV{ANDROID_NDK_MAJOR})
       if($ENV{ANDROID_NDK_MAJOR} VERSION_LESS "23")
         set_property(TARGET rtvm PROPERTY LINK_FLAGS -fuse-ld=gold)
         set_property(TARGET tvm_runner PROPERTY LINK_FLAGS -fuse-ld=gold)
       endif()
     endif()
   endif()
endif()

target_include_directories(rtvm PUBLIC "../../include")
target_link_libraries(rtvm PUBLIC tvm_ffi_header)
target_include_directories(tvm_runner_objs PUBLIC "../../include")
target_link_libraries(tvm_runner_objs PUBLIC tvm_ffi_header)

if (BUILD_FOR_ANDROID AND USE_HEXAGON)
  get_hexagon_sdk_property("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}"
    DSPRPC_LIB DSPRPC_LIB_DIRS
  )
  if(DSPRPC_LIB_DIRS)
    link_directories(${DSPRPC_LIB_DIRS})
  else()
    message(WARNING "Could not locate some Hexagon SDK components")
  endif()
  list(APPEND RTVM_LINKER_LIBS cdsprpc log)
endif()

# Device backends (OpenCL, CUDA, Vulkan, ...) are separate shared runtime libraries
# that register their DeviceAPI at load time, so they must stay linked even though
# rtvm references no symbol from them (hence --no-as-needed).
if(BUILD_STATIC_RUNTIME)
  foreach(lib tvm_runtime tvm_ffi_static)
    if(MSVC)
      list(APPEND RTVM_LINKER_LIBS "/WHOLEARCHIVE:$<TARGET_FILE:${lib}>")
    elseif(APPLE)
      list(APPEND RTVM_LINKER_LIBS "-Wl,-force_load,$<TARGET_FILE:${lib}>")
    else()
      list(APPEND RTVM_LINKER_LIBS "-Wl,--whole-archive" "${lib}" "-Wl,--no-whole-archive")
    endif()
  endforeach()
else()
  if(NOT MSVC AND NOT APPLE)
    list(APPEND RTVM_LINKER_LIBS tvm_runtime "-Wl,--no-as-needed" ${TVM_RUNTIME_BACKEND_LIBS} "-Wl,--as-needed")
  else()
    list(APPEND RTVM_LINKER_LIBS tvm_runtime ${TVM_RUNTIME_BACKEND_LIBS})
  endif()
endif()

target_link_libraries(rtvm PRIVATE ${RTVM_LINKER_LIBS})
target_link_libraries(tvm_runner PRIVATE ${RTVM_LINKER_LIBS})
