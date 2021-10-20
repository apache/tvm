cmake_policy(SET CMP0069 NEW) # suppress cmake warning about IPO

set(TVM_RPC_SOURCES
  main.cc
  rpc_env.cc
  rpc_server.cc
)

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
    set_property(TARGET tvm_rpc PROPERTY LINK_FLAGS -fuse-ld=gold)
  endif()
endif()

target_include_directories(
  tvm_rpc
  PUBLIC "../../include"
  PUBLIC DLPACK_PATH
  PUBLIC DMLC_PATH
)

target_link_libraries(tvm_rpc tvm_runtime)
