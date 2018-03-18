#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "../../src/runtime/sgx/trusted/runtime.h"
#include "../../src/runtime/sgx/trusted/runtime.cc"

using namespace tvm::runtime;

extern "C" {
void ecall_tvm_main(const void* args, const int* type_codes, int num_args) {
  Module mod = (*Registry::Get("module._GetSystemLib"))();
  PackedFunc f = mod.GetFunction("addonesys");
  TVMRetValue rv;
  const TVMValue* arg_values = reinterpret_cast<const TVMValue*>(args);
  f.CallPacked(TVMArgs(arg_values, type_codes, num_args), &rv);
}
}
