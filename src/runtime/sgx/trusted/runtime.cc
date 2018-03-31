/*!
 *  Copyright (c) 2018 by Contributors
 * \file runtime_t.cc
 */
#include "../../c_runtime_api.cc"
#include "../../cpu_device_api.cc"
#include "../../module.cc"
#include "../../module_util.cc"
#include "../../registry.cc"
#include "../../system_lib_module.cc"
#include "../../thread_pool.cc"
#include "../../workspace_pool.cc"
#include "threading_backend.cc"

using namespace tvm::runtime;

TVM_REGISTER_ENCLAVE_FUNC("__tvm_main__")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module mod = (*Registry::Get("module._GetSystemLib"))();
    mod.GetFunction("addonesys").CallPacked(args, rv);
  });
