if(NOT USE_SGX STREQUAL "OFF")

  set(_sgx_src ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/sgx)
  set(_tvm_u_h ${_sgx_src}/untrusted/tvm_u.h)
  set(_tvm_t_h ${_sgx_src}/trusted/tvm_t.h)
  set(_tvm_t_c ${_sgx_src}/trusted/tvm_t.c)
  set(_tvm_edl ${_sgx_src}/tvm.edl)
  set(_sgx_ustdc ${RUST_SGX_SDK}/sgx_ustdc)

  set(_urts_lib "sgx_urts")
  if(NOT SGX_MODE STREQUAL "HW")
    message(STATUS "Build with SGX support (SIM)")
    set(_urts_lib "${_urts_lib}_sim")
  else()
    message(STATUS "Build with SGX support (HW)")
  endif()

  # build edge routines
  add_custom_command(
    OUTPUT ${_tvm_u_h}
    COMMAND ${USE_SGX}/bin/x64/sgx_edger8r --untrusted
      --untrusted --untrusted-dir ${_sgx_src}/untrusted
      --trusted --trusted-dir ${_sgx_src}/trusted
      --search-path ${USE_SGX}/include --search-path ${RUST_SGX_SDK}/edl
      ${_tvm_edl}
    COMMAND sed -i "4i '#include <tvm/runtime/c_runtime_api.h>'" ${_tvm_u_h}
    COMMAND sed -i "4i '#include <tvm/runtime/c_runtime_api.h>'" ${_tvm_t_h}
    DEPENDS ${_tvm_edl}
  )
  add_custom_command(
    OUTPUT ${_sgx_ustdc}/libsgx_ustdc.a
    COMMAND make
    WORKING_DIRECTORY ${_sgx_ustdc}
  )
  add_custom_target(sgx_edl DEPENDS ${_tvm_u_h} ${_sgx_ustdc}/libsgx_ustdc.a)

  # build trusted library
  set_source_files_properties(${_tvm_t_c} PROPERTIES GENERATED TRUE)
  add_library(tvm_t STATIC ${_tvm_t_c})
  add_dependencies(tvm_t sgx_edl)
  target_include_directories(tvm_t PUBLIC ${USE_SGX}/include ${USE_SGX}/include/tlibc)

  # add untrusted runtime files
  include_directories(${USE_SGX}/include)
  file(GLOB RUNTIME_SGX_SRCS ${_sgx_src}/untrusted/*.c*)
  list(APPEND TVM_RUNTIME_LINKER_LIBS
    -lpthread
    -L${USE_SGX}/lib64 -l${_urts_lib}
    -L${RUST_SGX_SDK}/sgx_ustdc -lsgx_ustdc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_SGX_SRCS})
endif()
