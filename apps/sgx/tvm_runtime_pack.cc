/*!
 * \brief This is an all in one TVM runtime file for use in an SGX enclave.
 *
 *   The files included here will be statically linked into the enclave.
 *   Please refer to the Makefile (rule lib/tvm_runtime_pack.o) for how to build.
 *
 */
#ifdef _LIBCPP_SGX_CONFIG
#include "lib/test_addone_t.h"
#endif
#include "../../sgx/runtime_t.cc"

#ifndef _LIBCPP_SGX_CONFIG
#include "../../src/runtime/file_util.cc"
#endif
