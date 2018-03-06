/*!
 * \brief This is an all in one TVM runtime file for use in an SGX enclave.
 *
 *   The files included here will be statically linked into the enclave.
 *   Please refer to the Makefile (rule lib/tvm_runtime_pack.o) for how to build.
 *
 */
#include "../../sgx/sgx_runtime.cc"
#ifndef _LIBCPP_SGX_CONFIG
#include "../../src/runtime/file_util.cc"
#endif
