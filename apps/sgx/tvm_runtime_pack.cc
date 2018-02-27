/*!
 * \brief This is an all in one TVM runtime file for use in an SGX enclave.
 *
 *   The files included here will be statically linked into the enclave.
 *   Please refer to the Makefile (rule lib/tvm_runtime_pack.o) for how to build.
 *
 */
#include "../../src/runtime/c_runtime_api.cc"
#include "../../src/runtime/cpu_device_api.cc"
#include "../../src/runtime/workspace_pool.cc"
#include "../../src/runtime/module_util.cc"
#include "../../src/runtime/module.cc"
#include "../../src/runtime/registry.cc"
#include "../../src/runtime/system_lib_module.cc"
#ifndef _LIBCPP_SGX_CONFIG
#include "../../src/runtime/file_util.cc"
#endif
