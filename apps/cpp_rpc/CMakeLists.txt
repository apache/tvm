cmake_policy(SET CMP0069 NEW) # suppress CMake warning about IPO

set(TVM_RPC_SOURCES
  main.cc
  rpc_env.cc
  rpc_server.cc
)

set(TVM_RPC_LINKER_LIBS "")

if(WIN32)
  list(APPEND TVM_RPC_SOURCES win32_process.cc)
endif()

# Set output to same directory as the other TVM libs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(tvm_rpc ${TVM_RPC_SOURCES})

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET tvm_rpc PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(WIN32)
  target_compile_definitions(tvm_rpc PUBLIC -DNOMINMAX)
endif()

if (OS)
   if (OS STREQUAL "Linux")
      set_property(TARGET tvm_rpc PROPERTY LINK_FLAGS -lpthread)
   endif()
endif()

if(USE_OPENCL)
   if (ANDROID_ABI)
     if(DEFINED ENV{ANDROID_NDK_MAJOR})
       if($ENV{ANDROID_NDK_MAJOR} VERSION_LESS "23")
         set_property(TARGET tvm_rpc PROPERTY LINK_FLAGS -fuse-ld=gold)
       endif()
     endif()
   endif()
endif()

target_include_directories(
  tvm_rpc
  PUBLIC "../../include"
)

target_link_libraries(tvm_rpc PUBLIC tvm_ffi_header)

if (BUILD_FOR_ANDROID AND USE_HEXAGON)
  get_hexagon_sdk_property("${USE_HEXAGON_SDK}" "${USE_HEXAGON_ARCH}"
    DSPRPC_LIB DSPRPC_LIB_DIRS
  )
  if(DSPRPC_LIB_DIRS)
    link_directories(${DSPRPC_LIB_DIRS})
  else()
    message(WARNING "Could not locate some Hexagon SDK components")
  endif()
  list(APPEND TVM_RPC_LINKER_LIBS cdsprpc log)
endif()

if(BUILD_STATIC_RUNTIME)
  foreach(lib ${TVM_RUNTIME_BACKEND_LIBS})
    if(MSVC)
      list(APPEND TVM_RPC_LINKER_LIBS "/WHOLEARCHIVE:$<TARGET_FILE:${lib}>")
    elseif(APPLE)
      list(APPEND TVM_RPC_LINKER_LIBS "-Wl,-force_load,$<TARGET_FILE:${lib}>")
    else()
      list(APPEND TVM_RPC_LINKER_LIBS "-Wl,--whole-archive" "${lib}" "-Wl,--no-whole-archive")
    endif()
  endforeach()
else()
  if(NOT MSVC AND NOT APPLE)
    list(APPEND TVM_RPC_LINKER_LIBS tvm_runtime "-Wl,--no-as-needed" ${TVM_RUNTIME_BACKEND_LIBS} "-Wl,--as-needed")
  else()
    list(APPEND TVM_RPC_LINKER_LIBS tvm_runtime ${TVM_RUNTIME_BACKEND_LIBS})
  endif()
endif()

target_link_libraries(tvm_rpc PRIVATE ${TVM_RPC_LINKER_LIBS})
