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
set_property(TARGET tvm_rpc PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

if(WIN32)
  target_compile_definitions(tvm_rpc PUBLIC -DNOMINMAX)
endif()

target_include_directories(
  tvm_rpc
  PUBLIC "../../include"
  PUBLIC DLPACK_PATH
  PUBLIC DMLC_PATH
)
 
target_link_libraries(tvm_rpc tvm_runtime)